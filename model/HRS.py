import einops
import torch
from torch import nn

class FeatureFusionModule(nn.Module):
    def __init__(self, feature_concat_len, hidden_dim, pred_len, ffm_len):
        super(FeatureFusionModule, self).__init__()
        self.ffm_len = ffm_len
        self.linear_mix = nn.Linear(feature_concat_len, self.ffm_len)

    def forward(self, f_n, f_v):
        # Feature concatenation
        x = torch.cat((f_n, f_v), dim=1)    # [batch, fig_patch_num + seq_patch_num, dim]
        
        # Feature fusion
        x = torch.transpose(x, 1, 2)    # [batch, dim, fig_patch_num + seq_patch_num]
        x = self.linear_mix(x)  # [batch, dim, mul_len]
        x = torch.transpose(x, 1, 2)    # [batch, mul_len, dim]
        return x

class NumericalFeatureExtraction(nn.Module):
    def __init__(self, c_in, hidden_dim, freq, dropout=0.1):
        super(NumericalFeatureExtraction, self).__init__()
        
        # Value Information
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.value_information_extraction = nn.Conv1d(in_channels=c_in, out_channels=hidden_dim,
                                       kernel_size=3, padding=padding, 
                                       padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        # Timestamp Information
        freq_map = {'h': 4, 't': 5, 's': 6,
                   'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.timestamp_information_extraction = nn.Linear(d_inp, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_information_extraction(x.permute(0, 2, 1)).transpose(1, 2)
        x = x + self.timestamp_information_extraction(x_mark)
        return self.dropout(x)

class VisualFeatureExtraction(nn.Module):
    def __init__(self, channel, hidden_dim, patch_size, stride):
        super(VisualFeatureExtraction, self).__init__()
        self.conv_embedding = nn.Conv2d(channel, hidden_dim, stride=stride, 
                                      kernel_size=patch_size, padding=0)

    def forward(self, fig_x):
        f_m = torch.reshape(fig_x, (-1, fig_x.shape[1], fig_x.shape[3], fig_x.shape[2]))
        f_m = self.conv_embedding(f_m.float())
        f_v = einops.rearrange(f_m, 'b c h w -> b (h w) c')
        return f_v

class MultiDependenciesLearningModule(nn.Module):

    def __init__(self, hidden_dim, ffm_len, token_mlp_dim, dimension_mlp_dim, dropout, n_blocks):
        super(MultiDependenciesLearningModule, self).__init__()
        
        self.MDM_blocks = nn.ModuleList([
            MDMBlock(hidden_dim, ffm_len, token_mlp_dim, dimension_mlp_dim, dropout) 
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for MDM_block in self.MDM_blocks:
            x = MDM_block(x)
        return x
    
class MDMBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, token_mlp_dim, dimension_mlp_dim, dropout):
        super(MDMBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.mlp_token = MlpBlock(token_dim, token_mlp_dim, dropout)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp_dimension = MlpBlock(hidden_dim, dimension_mlp_dim, dropout)

    def forward(self, x):
        y = self.layer_norm_1(x)  # (bs, token, dimension)
        y = torch.transpose(y, -1, -2)  # (bs, dimension, token)
        y = self.mlp_token(y)
        y = torch.transpose(y, -1, -2)
        x = x + y  # (bs, token, dimension)
        y = self.layer_norm_2(x)
        y = self.mlp_dimension(y)
        x = x + y

        return x

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout):
        super(MlpBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.visual_feature_len = (
                ((args.seq_len - args.patch_size[0]) // args.stride[0] + 1) *
                ((args.h - args.patch_size[1]) // args.stride[1] + 1) 
        )
        self.num_feature_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.pred_len = args.pred_len
        self.ffm_len = args.ffm_len
        
        # Visual Feature Extraction
        self.visual_extraction = VisualFeatureExtraction(
            channel=args.channel,
            hidden_dim=self.hidden_dim,
            patch_size=args.patch_size,
            stride=args.stride
        )

        # Numerical Feature Extraction
        self.seq_embedding = NumericalFeatureExtraction(args.n_feature, self.hidden_dim, args.freq, args.dropout)

        self.feature_concat_len = self.visual_feature_len + self.num_feature_len

        # Feature Fusion
        self.feature_fusion = FeatureFusionModule(
            feature_concat_len=self.feature_concat_len,
            hidden_dim=self.hidden_dim,
            pred_len=self.pred_len,
            ffm_len=self.ffm_len
        )

        # Multi-Dependencies Learning
        self.multi_dependencies_learning = MultiDependenciesLearningModule(
            hidden_dim=self.hidden_dim,
            ffm_len=self.ffm_len,
            token_mlp_dim=args.token_mlp_dim,
            dimension_mlp_dim=args.dimension_mlp_dim,
            dropout=args.dropout,
            n_blocks=args.n_blocks
        )

        # Projection
        self.linear1 = nn.Linear(self.ffm_len, self.pred_len)
        self.linear2 = nn.Linear(self.hidden_dim, 1)

        self.args = args

    def forward(self, x_num, x_img, x_num_mark):
        # CI
        bc, seq_len, f = x_num.shape
        x_num = torch.transpose(x_num, 1, 2)
        x_num = torch.reshape(x_num, (-1, 1, seq_len))
        x_num = torch.transpose(x_num, 1, 2)
        
        # Visual Feature Extraction
        f_v = self.visual_extraction(x_img)
        
        # Numerical Feature Extraction
        f_n = self.seq_embedding(x_num, x_num_mark)

        # Feature Fusion
        f_f = self.feature_fusion(f_n, f_v)

        # Multi-Dependencies Learning
        x = self.multi_dependencies_learning(f_f)

        # Projection
        x = torch.transpose(x, 1, 2)
        x = self.linear1(x)
        x = torch.transpose(x, 1, 2)
        x = self.linear2(x)

        # de CIs
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (bc, f, self.pred_len))
        x = torch.transpose(x, 1, 2)
        return x.float()