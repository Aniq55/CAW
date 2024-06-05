import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN
from graph import NeighborFinder
import resource
from loguru import logger as loguru1


split_times = {
    "reddit": (1882469.265, 2261813.658),
    "Contacts": (1625100, 2047800),
    "wikipedia": (1862652.1, 2218288.6),
    "uci": (3834800.6, 6714558.3),
    "SocialEvo": (16988541, 18711359),
    "mooc": (1917235, 2250151.6),
    "lastfm": (103764807.2, 120235473),
    "enron": (83843725.6, 93431801),
    "Flights": (90, 106),
    "UNvote": (1830297600, 2019686400),
    "CanParl": (283996800, 347155200),
    "USLegis": (8, 10),
    "UNtrade": (757382400, 883612800)
}

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
DISTORTION = args.distortion
MODEL_NAME = "CAWN"
LOAD_MODEL = args.loadmodel

if DISTORTION!='':
    DISTORTION_A = DISTORTION.split('_')[0]
    DISTORTION_B = DISTORTION.split('_')[-2]
else:
    DISTORTION_A, DISTORTION_B = '', ''


# Define custom log levels
INFO1_LEVEL = 25
INFO2_LEVEL = 35

# Add custom levels to loguru
loguru1.level("INFO1", INFO1_LEVEL)
loguru1.level("INFO2", INFO2_LEVEL)


LOG_FILE_val = f"/home/chri6578/Documents/gttp/logs/evalcheck/{DATA}_val.log"
LOG_FILE_test = f"/home/chri6578/Documents/gttp/logs/evalcheck/{DATA}_test.log"

loguru1.add(LOG_FILE_val, level=INFO1_LEVEL, format="{message}", filter=lambda record: record["level"].name == "INFO1")
loguru1.add(LOG_FILE_test, level=INFO2_LEVEL, format="{message}", filter=lambda record: record["level"].name == "INFO2")


assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, MODEL_SAVE_PATH = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv(f'/home/chri6578/Documents/gttp/data/{DATA}/{DISTORTION}ml_{DATA}.csv')
if args.data_usage < 1:
    g_df = g_df.iloc[:int(args.data_usage*g_df.shape[0])]
    logger.info('use partial data, ratio: {}'.format(args.data_usage))
e_feat = np.load(f'/home/chri6578/Documents/gttp/data/{DATA}/{DISTORTION}ml_{DATA}.npy')
n_feat = np.load(f'/home/chri6578/Documents/gttp/data/{DATA}/ml_{DATA}_node.npy')
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1 or ~math.isclose(1, args.data_usage))  # the nodes need to map one-to-one to the node feat matrix

# split and pack the data by generating valid train/val/test mask according to the "mode"
# val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
# val_time , test_time = 1862652.1, 2218288.6 # wikipedia
val_time, test_time = split_times[DATA]


if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    # mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_node_set = set(random.sample( list(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time]))), int(0.1 * num_total_unique_nodes)))

    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
    valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
    # valid_test_flag_inductive = (valid_test_new_new_flag + valid_test_new_old_flag) > 1
    # valid_test_flag_transductive = valid_test_flag*(1-valid_test_flag_inductive) 
    logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
if args.mode == 'i':
    test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag]
    test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder

# create random samplers to generate train/val/test instances
train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
cawn = CAWN(n_feat, e_feat, agg=AGG,
            num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
            n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
            num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
            cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)

# LOAD MODEL HERE

cawn.load_state_dict(torch.load(MODEL_SAVE_PATH))
cawn = cawn.to(device)
cawn.eval()

cawn.to(device)
optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

# start train and val phases
# train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

## VALIDATION:
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data
if args.mode == 't':
    cawn.update_ngh_finder(full_ngh_finder)
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for t nodes', cawn, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
    # MODEL \t DISTORTION \t SAMPLE \t SPLIT \t TYPE \t ACC \t AUC \t AP \t EPOCH
    info1_message = [MODEL_NAME, DISTORTION_A, DISTORTION_B, "Val", "tdv", 
                    f"{val_acc:.4f}", f"{val_auc:.4f}", f"{val_ap:.4f}", f"{early_stopper.best_epoch}"]
    loguru1.log("INFO1", '\t'.join(info1_message))

elif args.mode == 'i':
    cawn.update_ngh_finder(partial_ngh_finder)
    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for t nodes', cawn, val_rand_sampler, val_src_l,
                                                            val_dst_l, val_ts_l, val_label_l, val_e_idx_l)
    info1_message = [MODEL_NAME, DISTORTION_A, DISTORTION_B, "Val", "idv", 
                    f"{nn_val_acc:.4f}", f"{nn_val_auc:.4f}", f"{nn_val_ap:.4f}", f"{early_stopper.best_epoch}"]
    loguru1.log("INFO1", '\t'.join(info1_message))



# final testing
cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l)
logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
if args.mode == 'i':
    test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
    logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_auc,test_new_new_ap ))
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)
    logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_auc, test_new_old_ap))

if args.mode =='i':
    # MODEL \t DISTORTION \t SAMPLE \t SPLIT \t TYPE \t ACC \t AUC \t AP
    info2_message = [MODEL_NAME, DISTORTION_A, DISTORTION_B, "Test", "idv",
                    f"{test_acc:.4f}", f"{test_auc:.4f}", f"{test_ap:.4f}"]
elif args.mode =='t': 
    info2_message = [MODEL_NAME, DISTORTION_A, DISTORTION_B, "Test", "tdv",
                    f"{test_acc:.4f}", f"{test_auc:.4f}", f"{test_ap:.4f}"]
    
    
loguru1.log("INFO2", '\t'.join(info2_message))