from easydict import EasyDict

__C = EasyDict()

# ===========================================

__C.train_csv_path = '../inputs/train.csv'
__C.test_csv_path = '../inputs/test.csv'
__C.pred_path = '../results/'
__C.single_model_pred_path = __C.pred_path + 'single_model/'
__C.prejudge_pred_path = __C.pred_path + 'prejudge/'
__C.stack_pred_path = __C.pred_path + 'stacking/'
__C.auto_train_pred_path = __C.pred_path + 'auto_train/'
__C.log_path = '../logs/'
__C.csv_log_path = __C.log_path + 'csv_logs/'
__C.grid_search_out_path = '../grid_search_outputs/'
__C.boost_round_out_path = '../boost_round_outputs/'
__C.loss_log_path = __C.log_path + 'loss_logs/'
__C.prejudge_loss_log_path = __C.loss_log_path + 'prejudge/'
__C.dnn_log_path = __C.log_path + 'dnn_logs/'
__C.grid_search_log_path = __C.log_path + 'grid_search_logs/'
__C.data_path = '../data/'
__C.preprocessed_path = __C.data_path + 'preprocessed_data/'
__C.gan_prob_path = __C.data_path + 'gan_outputs/'
__C.gan_preprocessed_data_path =  __C.data_path + 'gan_preprocessed_data/'
__C.preprocessed_data_path = __C.preprocessed_path
__C.prejudged_data_path = __C.data_path + 'prejudged_data/'
__C.stack_output_path = __C.data_path + 'stacking_outputs/'
__C.model_checkpoint_path = '../checkpoints/'
__C.dnn_checkpoint_path = __C.model_checkpoint_path + 'dnn_checkpoints/'
__C.tsne_outputs_path = __C.data_path + 'tsne_outputs/'

# ===========================================

# get config by: from config import cfg
cfg = __C
