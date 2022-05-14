import torch
import sys

import code.embedding_modules as em
import code.pooling_modules as pm
model_params = { 'device' : torch.device("cuda" ),
                 'sparse_grads' : True,
                 'label_hashing' : False,
                 'type_smoothing' : False,
                 'mask_rate' : 0.01,
                 'default_pc' : True,
                 'lr_decay' : False,
                 'att_aggr' : True,
                 'mha_heads' : 2,
                 'edge_type' : None,
                 'dep_gate' : True,
                 # learning rate
                 'lr' : 0.001,
                 'depth_cap' : None,
                 'aggr_type' : 'sum',
                 'dep_match_type' : 'label',
                 'dep_depth' : 0,
                 'dropout' : 0.2,#0.2,
                 # node info
                 'node_ct' : 30000,
                 'node_emb_dim' : int(128 / 4),#32 or 64
                 'node_state_dim' :  int(256 / 4),
                 'lstm_state_dim' : int(256 / 4),
                 'pretrained_emb_dim' : 50,
                 # edge info
                 'edge_ct' : 150,
                 'edge_emb_dim' : int(32 / 4),
                 'edge_state_dim' :  int(64 / 4),
                 # module used to get initial node embeddings
                 'init_node_embedder' : em.MPNN,
                 # if init node embedder is DagLSTM
                 'init_node_embedder_acc_dir' : em.ACCNN.down_acc,
                 # if init node embedder is GCN or MPNN
                 'num_rounds' : 0,
                 # pooling type used for graph embedding
                 'pooling_module' : pm.SimpleMaxPool,
                 # to use leaf or root pooling
                 'pooling_dir' : em.ACCNN.up_acc ,
                 'pretrained_embs':False}
