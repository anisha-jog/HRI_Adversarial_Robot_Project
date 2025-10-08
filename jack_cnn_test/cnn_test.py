# %% IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
# %% Create Dataset Class
class PartialImageDatasetPreLoad(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.partial_img_data = []
        self.partial_img_label = []
        self.partial_img_next = []
        blank_img = np.full((256, 256),255, dtype = np.uint8)
        for idx, drawing in enumerate(image_df['drawing']):
            img = blank_img.copy()
            strokes = json.loads(drawing)
            for stroke_num in range(len(strokes)-1):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
                self.partial_img_data.append(img.copy())
                self.partial_img_label.append(image_df['word'][idx])
                just_next_stroke = self.add_stroke_to_image(blank_img, strokes[stroke_num+1])
                self.partial_img_next.append(just_next_stroke)
        self.partial_img_data = np.array(self.partial_img_data)
        self.partial_img_label = np.array(self.partial_img_label)
        self.partial_img_next = np.array(self.partial_img_next)
        assert self.partial_img_data.shape[0] == self.partial_img_label.shape[0] == self.partial_img_next.shape[0]

    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.partial_img_data.shape[0]

    def __getitem__(self, idx):
        return self.partial_img_data[idx], self.partial_img_label[idx], self.partial_img_next[idx]

class PartialImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame):
        super().__init__()
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.cum_sum_strokes = []
        for drawing in (image_df['drawing']):
            strokes = json.loads(drawing)
            self.cum_sum_strokes.append(len(strokes)-1)
        self.cum_sum_strokes = np.cumsum(self.cum_sum_strokes)

    def get_subidx(self, idx):
        sample_idx = np.searchsorted(self.cum_sum_strokes, idx, side='right')
        if sample_idx == 0:
            stroke_idx = idx
        else:
            stroke_idx = idx - self.cum_sum_strokes[sample_idx-1]
        return sample_idx, stroke_idx+1


    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.cum_sum_strokes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            sample_idx, final_stroke_idx = self.get_subidx(idx)
            blank_img = np.full((256, 256),255, dtype = np.uint8)
            img = blank_img.copy()
            strokes = json.loads(self.raw_data['drawing'][sample_idx])
            for stroke_num in range(final_stroke_idx):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
            partial_img_data = img.copy()
            partial_img_label = self.raw_data['word'][sample_idx]
            just_next_stroke = self.add_stroke_to_image(blank_img, strokes[final_stroke_idx])

            return partial_img_data, partial_img_label, just_next_stroke

def to_device(data, device):
    """
    Recursively moves all tensors in a data structure (list, tuple, dict) to a specific device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data # Return non-tensor data as is

def raw_stroke_to_normed(stroke:np.ndarray,max_stroke_len:int):
    stroke_swap = np.swapaxes(stroke,1,0).astype(dtype=np.float32)
    mu_arr = stroke_swap.mean(axis=0)
    dxdy = stroke_swap[-1] - stroke_swap[0]
    # s = np.sqrt(np.square(stroke_swap.max(axis=0) - stroke_swap.min(axis=0)).sum())
    s = (stroke_swap.max(axis=0) - stroke_swap.min(axis=0)).max()
    s = np.maximum(s, 1e-6)
    r = np.sqrt(dxdy[0]**2 + dxdy[1]**2)
    r = np.maximum(r, 1e-6)
    cos_theta = dxdy[0] / r
    sin_theta = dxdy[1] / r
    normed_arr = np.zeros_like(stroke_swap)
    normed_arr[:,0] = ((stroke_swap[:,0] - mu_arr[0]) * cos_theta + (stroke_swap[:,1] - mu_arr[1]) * sin_theta) / s
    normed_arr[:,1] = ((stroke_swap[:,0] - mu_arr[0]) * sin_theta + (stroke_swap[:,1] - mu_arr[1]) * cos_theta) / s
    pad_len = max_stroke_len - stroke_swap.shape[0]
    padded_stroke = np.pad(normed_arr,((0,pad_len),(0,0)),mode='edge')

    eos_arr = np.zeros((max_stroke_len,),dtype=padded_stroke.dtype)
    eos_arr[stroke.shape[1]-1] += 1

    params_arr = np.array([mu_arr[0], mu_arr[1], s, cos_theta, sin_theta])

    return params_arr, padded_stroke, eos_arr

def transform_stroke(params_arr, padded_stroke, eos_arr):
    normed_stroke = padded_stroke[:int(np.where(eos_arr==1)[0][0])+1].copy()
    normed_stroke *= params_arr[2]
    stroke = np.zeros_like(normed_stroke)
    # stroke[:,0] = params_arr[0] + normed_stroke[:,0] * params_arr[3] - normed_stroke[:,0] * params_arr[4]
    # stroke[:,1] = params_arr[1] - normed_stroke[:,0] * params_arr[4] + normed_stroke[:,0] * params_arr[3]

    stroke[:,0] = params_arr[0] + (params_arr[3] * normed_stroke[:,0] - params_arr[4] * normed_stroke[:,1]) / (params_arr[3]**2 - params_arr[4]**2)
    stroke[:,1] = params_arr[1] + (params_arr[4] * normed_stroke[:,0] - params_arr[3] * normed_stroke[:,1]) / (params_arr[4]**2 - params_arr[3]**2)
    return stroke

class PartialImageStrokeDataset(torch.utils.data.Dataset):
    def __init__(self, image_df : pd.DataFrame, max_stroke_len:int):
        super().__init__()
        self.label_list = image_df['word'].unique().tolist()
        self.raw_data = image_df
        self.stroke_width = 1
        self.stroke_len = max_stroke_len
        self.cum_sum_strokes = []
        for drawing in (image_df['drawing']):
            strokes = json.loads(drawing)
            self.cum_sum_strokes.append(len(strokes)-1)
        self.cum_sum_strokes = np.cumsum(self.cum_sum_strokes)
        print(f"Strokes forced to length of {self.stroke_len}")

    def get_subidx(self, idx):
        sample_idx = np.searchsorted(self.cum_sum_strokes, idx, side='right')
        if sample_idx == 0:
            stroke_idx = idx
        else:
            stroke_idx = idx - self.cum_sum_strokes[sample_idx-1]
        return sample_idx, stroke_idx+1


    def add_stroke_to_image(self, image, stroke):
        for i in range(len(stroke[0])-1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), (0,0,0), self.stroke_width)
        return image

    def __len__(self):
        return self.cum_sum_strokes[-1]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        else:
            sample_idx, final_stroke_idx = self.get_subidx(idx)
            blank_img = np.full((256, 256),255, dtype = np.uint8)
            img = blank_img.copy()
            strokes = json.loads(self.raw_data['drawing'][sample_idx])
            for stroke_num in range(final_stroke_idx):
                img = self.add_stroke_to_image(img, strokes[stroke_num])
            partial_img_data = img.copy().astype(np.float32)
            partial_img_label = self.raw_data['word'][sample_idx]
            final_stroke = np.array(strokes[final_stroke_idx])
            stroke_params, stroke_normed, stroke_eos = raw_stroke_to_normed(final_stroke,max_stroke_len=self.stroke_len)

            return partial_img_data, partial_img_label, stroke_params, stroke_normed, stroke_eos

# %% Define the Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        # Only add dropout if the rate is non-zero
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(p=dropout_rate))
            # Note: Dropout2d is the correct choice for Conv layers

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

# --- 2. The Conditional U-Net Model ---
class ArtistModel(nn.Module):
    def __init__(self, img_channels=1, base_c=64,depth = 4, dropout_rate=.2, bert_model_name = 'bert-base-uncased',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.depth=depth
        self.device = device
        # Define channel sizes
        self.c_levels = [base_c*(2**i) for i in range(depth)]

        # Create Encoder
        self.encoders = nn.ModuleList([ConvBlock(img_channels if d==0 else self.c_levels[d-1], self.c_levels[d],dropout_rate=dropout_rate) for d in range(depth)])
        self.poll = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. BOTTLENECK (Feature Fusion Point)
        # The input to the bottleneck comes from the image (self.c_levels[-1]) plus the expanded language embedding (embed_dim).
        self.bottleneck = ConvBlock(self.c_levels[-1] + self.embed_dim, self.c_levels[-1],dropout_rate=dropout_rate*2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for d in range(depth-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(self.c_levels[d], self.c_levels[d-1], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(self.c_levels[d-1]+self.c_levels[d], self.c_levels[d-1]))  # times 2 for skip connection

        self.upconvs.append(nn.ConvTranspose2d(self.c_levels[0], self.c_levels[0], kernel_size=2, stride=2))
        self.decoders.append(ConvBlock(self.c_levels[0]*2, self.c_levels[0]))

        self.out_conv = nn.Conv2d(self.c_levels[0] , img_channels, kernel_size=1)

        self.to(self.device)

    @staticmethod
    def to_bw_img(img:torch.Tensor):
        return (img > 0.5).float() * 255

    def to_float_img(self,img:torch.Tensor):
        return img.unsqueeze(1).float().to(self.device) / 255.0

    def forward(self, x, text_input):        # x: (B, 1, 256, 256), lang_embedding: (B, E_DIM)
        c_inner = []
        for enc in self.encoders:
            x = enc(x)
            c_inner.append(x)
            x = self.poll(x)
        encoded_img = x

        tokenized_input = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Ensure BERT tensors are on the same device as the image
        tokenized_input = {k: v.to(x.device) for k, v in tokenized_input.items()}

        # Get BERT output (no gradient tracking for initial BERT output)
        bert_output = self.bert_model(**tokenized_input)

        # Extract the contextual embedding for the whole sentence (the [CLS] token)
        # embedding_vector shape: (B, EMBEDDING_DIM=768)
        embedding_vector = bert_output.last_hidden_state[:, 0, :]

        # a. Prepare the Language Embedding for Concatenation
        # Reshape language vector to match feature map dimensions
        B, C, H_f, W_f = encoded_img.shape
        # Expand: (B, E_DIM) -> (B, E_DIM, 1, 1) -> (B, E_DIM, H_f, W_f)
        lang_feature_map = embedding_vector[:, :, None, None].expand(B, self.embed_dim, H_f, W_f)
        # b. Concatenate Image Features and Language Features
        fused_features = torch.cat([encoded_img, lang_feature_map], dim=1) # (B, 512 + E_DIM, 16, 16)
        # c. Bottleneck Convolution
        b = self.bottleneck(fused_features) # (B, 512, 16, 16)

        # --- 3. DECODER ---
        for i in range(self.depth):
            u = self.upconvs[i](b)
            skip_idx = self.depth - 1 - i
            c = c_inner[skip_idx]
            # CROP/PAD SAFETY CHECK (Still important)
            if u.shape[-2:] != c.shape[-2:]:
                diffY = c.size()[2] - u.size()[2]
                diffX = c.size()[3] - u.size()[3]
                c = c[:, :, diffY//2:c.size()[2] - (diffY - diffY//2), diffX//2:c.size()[3] - (diffX - diffX//2)]

            u = torch.cat([u, c], dim=1)
            b = self.decoders[i](u)

        # --- 4. OUTPUT ---
        output_image = torch.sigmoid(self.out_conv(b))

        return output_image

# --- Attention Mechanism for LSTM Context ---
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        # W_h: Projects encoder features
        self.W_h = nn.Linear(encoder_dim, decoder_dim, bias=False)
        # W_s: Projects decoder hidden state
        self.W_s = nn.Linear(decoder_dim, decoder_dim, bias=False)
        # V: Projects combined energy to a single score
        self.V = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_features, decoder_hidden):
        # encoder_features: [B, H*W, C]
        # decoder_hidden: [B, D]

        # 1. Expand decoder state for broadcast addition: [B, 1, D]
        hidden_expanded = decoder_hidden.unsqueeze(1)

        # 2. Calculate Energy: tanh(W_h*Context + W_s*Hidden) -> [B, H*W, D]
        # We need to reshape encoder_features for the linear layer
        energy = torch.tanh(self.W_h(encoder_features) + self.W_s(hidden_expanded))

        # 3. Calculate Attention Scores: V * energy -> [B, H*W]
        scores = self.V(energy).squeeze(2)

        # 4. Normalize Scores (Softmax over spatial locations)
        alphas = F.softmax(scores, dim=1) # [B, H*W]

        # 5. Compute Context Vector: Weighted sum of context features -> [B, C]
        context_vector = (alphas.unsqueeze(2) * encoder_features).sum(dim=1)

        return context_vector, alphas

class StrokeModel(nn.Module):
    def __init__(self, base_c=64, depth=3, hidden_size=512,lang_hidden_size=128, dropout_p=0.2, max_stroke_len=100, img_length=256,
                 bert_model_name = 'bert-base-uncased',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        img_channels = 1
        self.img_len = img_length
        self.hidden_size = hidden_size
        self.lang_hidden_size = lang_hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device if hasattr(self, 'device') else 'cpu')
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.bert_model.config.hidden_size
        self.bert_projection = nn.Linear(self.embed_dim, self.lang_hidden_size)

        # --- U-NET ENCODER (Image Feature Extraction) ---
        self.depth = depth
        self.c_levels = [base_c * (2**d) for d in range(depth)]

        # Encoders are standard ConvBlocks with downsampling
        self.encoders = nn.ModuleList([
            # Assuming you reuse the ConvBlock from the previous U-Net structure
            ConvBlock(img_channels if d == 0 else self.c_levels[d - 1], self.c_levels[d], dropout_rate=dropout_p)
            for d in range(depth)
        ])

        # Downsampling layers
        self.downs = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(depth)
        ])

        # Bottleneck (Feature Fusion with Text/BERT, assuming BERT features are [768, 1, 1])
        # We need to broadcast and concatenate the BERT features to the image features
        self.bottleneck_c = self.c_levels[-1]
        self.bottleneck_conv = ConvBlock(self.bottleneck_c + self.embed_dim, self.bottleneck_c, dropout_rate=dropout_p + 0.1)

        # --- HEADS (Split into 2 tasks) ---

        # 1. PARAMETER HEAD (MLP for Transformation Parameters)
        # Output: [mu_x, mu_y, S, cos(theta), sin(theta)] = 5 parameters
        # Note: Using sin/cos is better for regression than a raw angle theta
        # Input features: Flattened bottleneck features (bottleneck_c * 4*4 if last HxW=4)
        final_h_w = int(self.img_len // (2 ** self.depth))
        self.param_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bottleneck_c * final_h_w * final_h_w, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 5) # 5 outputs: mu_x, mu_y, S, cos(theta), sin(theta)
        )

        # 2. SEQUENCE DECODER HEAD (LSTM for Normalized Coordinates)
        self.lstm_hidden_size = hidden_size

        # Initial hidden state (h0, c0) from bottleneck features
        # Bottleneck features are projected down to the size of the LSTM hidden state
        self.h_init = nn.Linear(self.bottleneck_c * final_h_w * final_h_w + self.embed_dim, hidden_size)
        self.c_init = nn.Linear(self.bottleneck_c * final_h_w * final_h_w + self.embed_dim, hidden_size)

        # Attention Mechanism (Context from bottleneck, Hidden from LSTM)
        self.attention = Attention(encoder_dim=self.bottleneck_c, decoder_dim=hidden_size)

        # LSTM Decoder (Input: previous coordinate (2) + context vector (hidden_size))
        self.lstm = nn.LSTM(input_size=2 + self.bottleneck_c + self.lang_hidden_size,
                            hidden_size=hidden_size,
                            batch_first=True)

        # Output layer: 2 coordinates (x, y) + 1 EOS logit
        self.output_layer = nn.Linear(hidden_size, 3)

        # Store max length for inference
        self.max_stroke_len = max_stroke_len
        self.device = device
        self.to(device)

    def forward(self, img_x, text_input):
        # 1. ENCODER PASS
        skips = []
        x = img_x
        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.downs[i](x)

        tokenized_input = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Ensure BERT tensors are on the same device as the image
        tokenized_input = {k: v.to(x.device) for k, v in tokenized_input.items()}

        # Get BERT output (no gradient tracking for initial BERT output)
        bert_output = self.bert_model(**tokenized_input)

        # Extract the contextual embedding for the whole sentence (the [CLS] token)
        # embedding_vector shape: (B, EMBEDDING_DIM=768)
        embedding_vector = bert_output.last_hidden_state[:, 0, :]
        projected_bert = self.bert_projection(embedding_vector)

        # 2. BOTTLENECK (Feature Fusion)
        # Bert embeds are typically [B, 768]. Reshape to [B, 768, 1, 1] and tile.
        bert_broadcast = embedding_vector.view(embedding_vector.size(0), -1, 1, 1).repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, bert_broadcast], dim=1)
        context : torch.Tensor = self.bottleneck_conv(x)

        # 3. SPLIT HEADS

        # Parameter Head
        flat_features = context.view(context.size(0), -1)
        flat_features_with_embedd = torch.cat([flat_features, embedding_vector], dim=1)
        param_output = self.param_head(context)

        # LSTM Initial States
        h0 = torch.tanh(self.h_init(flat_features_with_embedd)).unsqueeze(0) # [1, B, H]
        c0 = torch.tanh(self.c_init(flat_features_with_embedd)).unsqueeze(0) # [1, B, H]

        # Context Features for Attention (Reshaped: [B, H*W, C])
        attn_context_features = context.view(context.size(0), self.bottleneck_c, -1).transpose(1, 2)

        # Returns the predicted parameters and the context needed for the Sequence Decoding loop
        return param_output, h0, c0, attn_context_features, projected_bert

# %% --- LOSS FUNCTIONS ---
def create_gaussian_heatmap(coords, resolution=64, sigma=0.02):
    """
    Creates a differentiable heatmap from a sequence of normalized coordinates,
    treating the stroke as a series of connected line segments.
    Coords are expected to be in the normalized [-1, 1] space.

    Args:
        coords (torch.Tensor): Normalized coordinates [B, L, 2].
        resolution (int): HxW resolution of the output heatmap (e.g., 64).
        sigma (float): Standard deviation of the Gaussian kernel (controls smoothness/thickness).

    Returns:
        torch.Tensor: Heatmap tensor [B, 1, H, W].
    """
    device = coords.device
    B, L, _ = coords.shape

    # Check if there are enough points for segments (L >= 2)
    if L < 2:
        # If not enough points, return an empty heatmap (zero loss)
        return torch.zeros(B, 1, resolution, resolution, device=device)

    # 1. Create a coordinate grid for the output image (H x W)
    # Grid coordinates range from [-1, 1]
    xv, yv = torch.meshgrid([torch.linspace(-1, 1, resolution, device=device)] * 2, indexing='ij')
    grid = torch.stack([yv, xv], dim=-1) # [H, W, 2]

    # Grid expanded for broadcasting: [1, 1, H, W, 2]
    grid_expanded = grid.unsqueeze(0).unsqueeze(1)

    # 2. Define Line Segments (P1 and P2)
    P1 = coords[:, :-1].unsqueeze(2).unsqueeze(3) # Start points [B, L-1, 1, 1, 2]
    P2 = coords[:, 1:].unsqueeze(2).unsqueeze(3)  # End points [B, L-1, 1, 1, 2]

    # 3. Vector Calculation
    V = P2 - P1         # Segment vector (P1 -> P2) [B, L-1, 1, 1, 2]
    W = grid_expanded - P1 # Vector (P1 -> Grid Point) [B, L-1, H, W, 2]

    # V dot V (Squared length of the segment) [B, L-1, 1, 1]
    V_sq = torch.sum(V * V, dim=-1)

    # W dot V (Projection parameter numerator) [B, L-1, H, W]
    W_dot_V = torch.sum(W * V, dim=-1)

    # 4. Find the closest point on the segment
    # t is the scalar projection parameter for the infinite line
    # Clamp V_sq to prevent division by zero, but rely on geometric logic below
    t = W_dot_V / V_sq.clamp(min=1e-8)

    # Clamp t to [0, 1] to ensure the closest point is on the SEGMENT
    t_clamped = torch.clamp(t, 0.0, 1.0) # [B, L-1, H, W]

    # Closest Point (Pc) on the segment P1P2: Pc = P1 + t_clamped * V
    t_expanded = t_clamped.unsqueeze(-1) # [B, L-1, H, W, 1]
    Pc = P1 + t_expanded * V             # [B, L-1, H, W, 2]

    # 5. Calculate Squared Distance to the Closest Point
    # dist_sq = ||Grid Point - Closest Point||^2
    dist_sq = torch.sum((grid_expanded - Pc) ** 2, dim=-1) # [B, L-1, H, W]

    # 6. Calculate Gaussian weights
    # Gaussian formula: exp(-dist_sq / (2 * sigma^2))
    # Note: We sum over L-1 segments, not L points
    gaussian_maps = torch.exp(-dist_sq / (2 * sigma**2)) # [B, L-1, H, W]

    # 7. Aggregate all segments
    heatmap = torch.sum(gaussian_maps, dim=1) # [B, H, W]

    # 8. Normalize and Finalize
    # Clip values above 1.0
    heatmap = torch.clamp(heatmap, 0, 1)

    return heatmap.unsqueeze(1) # [B, 1, H, W]

class SequenceLossSmoothed(nn.Module):
    """
    Replaces point-wise coordinate loss with a smoothed image similarity loss (MSE
    on Gaussian heatmaps) while retaining the critical EOS classification loss.
    """
    def __init__(self, resolution=64, sigma=0.02):
        super(SequenceLossSmoothed, self).__init__()
        # Use MSE for image similarity (smoothed loss)
        self.img_criterion = nn.MSELoss()
        # Standard BCE for the classification task of terminating the stroke
        self.eos_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.resolution = resolution
        self.sigma = sigma

    def forward(self, pred_coords, pred_eos_logits, target_coords, target_eos):
        """
        Args:
            pred_coords: [B, L, 2] (Predicted normalized x, y)
            pred_eos_logits: [B, L] (Predicted EOS logit)
            target_coords: [B, L, 2] (Ground truth normalized x, y)
            target_eos: [B, L] (Ground truth EOS flag: 1.0 at EOS, 0.0 otherwise)
        """

        # 1. IMAGE/SHAPE LOSS (Smoothed Loss)
        # Use soft_mask_k in the heatmap generation
        pred_heatmap = create_gaussian_heatmap(pred_coords, self.resolution, self.sigma)
        target_heatmap = create_gaussian_heatmap(target_coords, self.resolution, self.sigma)

        # Calculate image similarity loss
        img_loss = self.img_criterion(pred_heatmap, target_heatmap)

        # 2. EOS Loss (Classification)
        eos_loss = self.eos_criterion(pred_eos_logits, target_eos)
        eos_loss = eos_loss.mean()

        # Returns the smoothed shape loss and the EOS loss
        return img_loss, eos_loss

class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()
        # Use L1 Loss (MAE) for coordinates for robustness against outliers
        self.coord_criterion = nn.L1Loss(reduction='none')
        self.eos_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_coords, pred_eos_logits, target_coords, target_eos):
        """
        Calculates loss for coordinates and EOS, using a mask to ignore padded steps.
        Args:
            pred_coords: [B, L, 2] (Predicted normalized x, y)
            pred_eos_logits: [B, L] (Predicted EOS logit)
            target_coords: [B, L, 2] (Ground truth normalized x, y)
            target_eos: [B, L] (Ground truth EOS flag: 1.0 at EOS, 0.0 otherwise)
        """

        # We only calculate coordinate loss up to the actual end of the stroke (or max length)
        # valid_mask is 1.0 for all steps *before* and *at* the EOS token (inclusive of EOS prediction)
        # Note: If target_eos has 1.0 at L, then the sequence length is L.

        # Mask calculation: Find the true length of each sequence
        # We use target_coords.sum(dim=2) to identify where padding starts (sum is 0).
        # We use 1.0 for valid points and 0.0 for padding.
        # Assuming padding in target_coords is 0, 0
        valid_length_mask = (target_coords.sum(dim=2) != -2).float()
        # This simple mask works if (0, 0) is not a common normalized coordinate.

        # 1. Coordinate Loss (MAE)
        coord_loss_unmasked = self.coord_criterion(pred_coords, target_coords).sum(dim=2) # [B, L]
        coord_loss = (coord_loss_unmasked * valid_length_mask).sum() / valid_length_mask.sum().clamp(min=1e-6)

        # 2. EOS Loss (BCE)
        # We train EOS loss over the entire sequence including padding,
        # but the coordinate loss only covers the actual stroke.
        eos_loss = self.eos_criterion(pred_eos_logits, target_eos).mean()

        return coord_loss, eos_loss

# --- TEACHER FORCING DECODER HELPER ---

def decode_lstm_sequence(model, h0, c0, context_features, text_context, target_coords):
    """
    Decodes the sequence using Teacher Forcing (passing ground truth coordinates
    as input to the next time step).

    Args:
        model: The StrokeGeneratorModel instance.
        h0, c0: Initial hidden states [1, B, H].
        context_features: [B, H*W, C] (Encoded image features).
        target_coords: [B, L, 2] (Target normalized coordinates used for input).

    Returns:
        pred_coords: [B, L, 2]
        pred_eos_logits: [B, L]
    """
    B, L, _ = target_coords.size()

    # Initialize output storage
    pred_coords = torch.zeros(B, L, 2).to(target_coords.device)
    pred_eos_logits = torch.zeros(B, L).to(target_coords.device)

    # LSTM states (h_t, c_t) are initially (h0[0], c0[0]) since h0/c0 are [1, B, H]
    hx, cx = h0.squeeze(0), c0.squeeze(0)

    # The first input (t=0) is the start token (0, 0)
    # We loop from t=0 to L-1 (L steps)
    input_coord = torch.zeros(B, 2).to(target_coords.device)

    for t in range(L):
        # 1. Attention (hx is the current hidden state [B, H])
        context_vector, _ = model.attention(context_features, hx) # [B, C]

        # 2. LSTM Input: Concatenate current coordinate and context vector
        lstm_input = torch.cat([input_coord, context_vector,text_context], dim=1).unsqueeze(1) # [B, 1, 2+C+H]

        # 3. LSTM Step: (h_t, c_t) = LSTM(input, (h_{t-1}, c_{t-1}))
        lstm_output, (hx_t, cx_t) = model.lstm(lstm_input, (hx.unsqueeze(0), cx.unsqueeze(0)))

        # Update states for next step
        hx, cx = hx_t.squeeze(0), cx_t.squeeze(0)

        # 4. Final Output: [B, 3] (x, y, EOS logit)
        output = model.output_layer(lstm_output.squeeze(1))

        # Store predictions
        next_coord_raw = output[:, :2] # Raw (x, y) prediction
        pred_coords[:, t, :] = torch.tanh(next_coord_raw) # Bounded (x, y)
        # pred_coords[:, t, :] = output[:, :2]
        pred_eos_logits[:, t] = output[:, 2]

        # Teacher Forcing: The next input coordinate is the ground truth for this step
        if t < L - 1:
            input_coord = target_coords[:, t, :] # Use the ground truth coordinate

    return pred_coords, pred_eos_logits

class TVLoss(nn.Module):
    """
    Calculates the Total Variation Loss to encourage spatial smoothness.
    Penalizes the difference between adjacent pixels.
    """
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        # x is the model output (e.g., [B, C, H, W] in the [0, 1] range)

        # 1. Horizontal variation (difference between pixel (i, j) and (i, j+1))
        # We slice the tensor to align pixels for subtraction.
        # This calculates the differences for all but the last column.
        horiz_diff = x[:, :, :, :-1] - x[:, :, :, 1:]

        # 2. Vertical variation (difference between pixel (i, j) and (i+1, j))
        # This calculates the differences for all but the last row.
        vert_diff = x[:, :, :-1, :] - x[:, :, 1:, :]

        # Calculate the L2-norm squared for horizontal and vertical differences
        # We use squaring (L2) instead of absolute value (L1) as it is differentiable
        # and often results in better optimization.
        tv_loss = torch.sum(horiz_diff.pow(2)) + torch.sum(vert_diff.pow(2))

        # Scale by the weight
        return self.weight * tv_loss

class DiceLoss(nn.Module):
    """
    Computes 1 - Dice Score, which is (1 - 2*Intersection / (A + B)).
    Assumes input 'pred' is the probability output (after Sigmoid).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten tensors to operate on pixels, not batches/channels
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate Intersection (True Positives)
        intersection = (prediction * target).sum()

        # Calculate Union (Sum of all predicted and target pixels)
        dice_sum = prediction.sum() + target.sum()

        # Dice Score: 2 * Intersection / Union
        dice_score = (2. * intersection + self.smooth) / (dice_sum + self.smooth)

        # Dice Loss: 1 - Dice Score
        return 1. - dice_score


# %% Load Data
dataset_path = "data/master_doodle_dataframe.csv"
full_raw_data = pd.read_csv(dataset_path)
full_data = full_raw_data.drop(columns=["countrycode", "recognized", "key_id", "image_path"])
full_label_list = full_data['word'].unique().tolist()
# %% Subset Data
# subset_labels = ['apple', 'banana', 'bicycle', 'car', 'cat']
subset_labels = ['apple', 'cat']
subsample_dataset_ratio = 0.1
train_test_split_ratio = .8
split_seed = 42
batch_size = 4
data = full_data[full_data['word'].isin(subset_labels)].reset_index(drop=True)
data = data.sample(frac=subsample_dataset_ratio, random_state=split_seed).reset_index(drop=True)
max_stroke_len_arr = np.array([max([len(s[0]) for s in json.loads(d)]) for d in data['drawing']])
stroke_len = max_stroke_len_arr.max()
train_df = data.sample(frac=train_test_split_ratio, random_state=split_seed)
test_df = data.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
# train_set = PartialImageDataset(train_df)
# test_set = PartialImageDataset(test_df)
train_set = PartialImageStrokeDataset(train_df,max_stroke_len=stroke_len)
test_set = PartialImageStrokeDataset(test_df,max_stroke_len=stroke_len)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
print(f"Using subset labels: {subset_labels} and only {subsample_dataset_ratio*100}% of full dataset")
print(f"Train-Test split is: {train_test_split_ratio}")
# %% Build Stroke Model
model_depth = 3
input_img_size = 256
learning_rate = 1e-4
num_epochs = 3
# Loss Weights: Tune these carefully. Parameter loss often needs a higher weight.
lambda_seq_coord = 1.0  # Weight for Coordinate Loss (MAE)
lambda_seq_eos = 0.25    # Weight for EOS Loss (BCE)
lambda_params = 10.0    # Weight for Transformation Parameter Loss (MSE) - Crucial for reconstruction
seq_loss_res = 64
seq_loss_sig = .025
size_lang_hidden = 32

# --- INSTANTIATE MODEL AND OPTIMIZER ---
# Assuming BERT embeddings are 768 dim, matching StrokeGeneratorModel's default
model = StrokeModel(img_length=input_img_size, depth=model_depth, lang_hidden_size=size_lang_hidden ,max_stroke_len=stroke_len)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- LOSS FUNCTIONS ---
# criterion_sequence = SequenceLoss()
criterion_sequence = SequenceLossSmoothed(seq_loss_res, seq_loss_sig)
criterion_params = nn.MSELoss()

print(f"Model initialized on device: {model.device}")
print("Starting Training Loop (Image-to-Sequence)")

# --- NOTE ON DATALOADER ---
# Your Dataloader must yield the following tensors:
# 1. img_x:            [B, C, H, W] (Input image)
# 2. bert_embeds:      [B, E] (Text embedding, e.g., 768)
# 3. target_params:    [B, 5] (mu_x, mu_y, S, cos(theta), sin(theta))
# 4. target_coords:    [B, L, 2] (Normalized x, y coordinates, L = max_stroke_len)
# 5. target_eos:       [B, L] (1.0 at EOS, 0.0 otherwise)

# %% Train Stroke Model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # --- SIMULATED LOOP START (Replace with actual dataloader iteration) ---
    for i, batch in enumerate(train_dataloader):
        # 1. Load and move data (SIMULATION)
        img_x, labels, target_params, target_coords, target_eos = to_device(batch,model.device)
        img_x = img_x.unsqueeze(1)


        optimizer.zero_grad()

        # 2. Forward Pass (Encoder + Initial LSTM states)
        pred_params, h0, c0, context_features, text_context = model(img_x, labels)

        # 3. Parameter Loss (mu, S, theta components)
        loss_params = criterion_params(pred_params, target_params)

        # 4. Sequence Decoding (Teacher Forcing uses target_coords as input)
        pred_coords, pred_eos_logits = decode_lstm_sequence(model, h0, c0, context_features,text_context, target_coords)

        # 5. Sequence Loss (Coordinate + EOS)
        loss_coord, loss_eos = criterion_sequence(pred_coords, pred_eos_logits, target_coords, target_eos)

        # 6. Combine Losses
        total_loss = (lambda_params * loss_params) + \
                     (lambda_seq_coord * loss_coord) + \
                     (lambda_seq_eos * loss_eos)

        # 7. Backward Pass and Step
        total_loss.backward()
        # Optional: Clip gradients to prevent exploding gradients common in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += total_loss.item()

        if (i+1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {avg_loss:.4f} "
                  f"[P: {loss_params.item():.4f}, C: {loss_coord.item():.4f}, E: {loss_eos.item():.4f}]")
            running_loss = 0.0

    # --- Validation Loop ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # 1. Load and move data (SIMULATION)
            img_x, labels, target_params, target_coords, target_eos = to_device(batch,model.device)
            img_x = img_x.unsqueeze(1)


            pred_params, h0, c0, context_features, text_context = model(img_x, labels)
            pred_coords, pred_eos_logits = decode_lstm_sequence(model, h0, c0, context_features,text_context, target_coords)

            loss_params = criterion_params(pred_params, target_params)
            loss_coord, loss_eos = criterion_sequence(pred_coords, pred_eos_logits, target_coords, target_eos)

            total_loss = (lambda_params * loss_params) + \
                         (lambda_seq_coord * loss_coord) + \
                         (lambda_seq_eos * loss_eos)

            val_loss += total_loss.item()

    val_loss /= len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

# %% Build ArtistModel
model_depth = 3
learning_rate = 1e-3
white_ratio = .95
pos_weight_value = (1-white_ratio) / white_ratio
num_epochs = 3
lambda_bin = 0.01
lambda_dice = 2  # Weight for Dice Loss (Good starting point: 0.05 to 0.2)
lambda_tv = 1   # Weight for TV Loss (Start small: 0.005 to 0.05)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = ArtistModel(depth=model_depth)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion_bse = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))
criterion_dice = DiceLoss()
criterion_tv = TVLoss(weight=1.0)
# %% Train ArtistModel
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (partial_imgs, labels, next_strokes) in enumerate(train_dataloader):
        partial_imgs = model.to_float_img(partial_imgs)
        next_strokes = model.to_float_img(next_strokes)
        optimizer.zero_grad()
        outputs = model(partial_imgs, labels)
        loss_bce  = criterion_bse(outputs, next_strokes)
        loss_dice = criterion_dice(outputs, next_strokes)
        loss_tv = criterion_tv(outputs)
        loss_binary = 4 * torch.mean(outputs * (1 - outputs))
        total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {running_loss/100:.4f}")
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for partial_imgs, labels, next_strokes in test_dataloader:
            partial_imgs = model.to_float_img(partial_imgs)
            next_strokes = model.to_float_img(next_strokes)
            outputs = model(partial_imgs, labels)
            loss_bce  = criterion_bse(outputs, next_strokes)
            loss_dice = criterion_dice(outputs, next_strokes)
            loss_tv = criterion_tv(outputs)
            loss_binary = 4 * torch.mean(outputs * (1 - outputs))
            total_loss = loss_bce + lambda_bin * loss_binary + (lambda_dice * loss_dice) + (lambda_tv * loss_tv)
            val_loss += total_loss.item()
    val_loss /= len(test_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

# %% Save Model
import time
torch.save(model.state_dict(), f"../jack_cnn_test/saved_models/{time.strftime('%m%d_%H%M%S', time.localtime())}_artist_model.pth")

#%%
weights = torch.load("../jack_cnn_test/saved_models/1006_212659_artist_model.pth", weights_only=True,map_location=device)
model_test = ArtistModel(depth=model_depth).to(device)
model_test.load_state_dict(weights)
partial_imgs, labels, next_strokes =  next(iter(test_dataloader))
partial_imgs = partial_imgs.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
next_strokes = next_strokes.unsqueeze(1).float().to(device) / 255.0 * 2 - 1
outputs = model_test(partial_imgs, labels)
partial_imgs, labels, next_strokes, s =  next(iter(test_dataloader))
outputs = model(model.to_float_img(partial_imgs),labels)
o_img = (outputs > 0.5).squeeze().cpu().detach().float().numpy() * 255

# %%
@torch.no_grad()
def inference_decode_stroke(model:StrokeModel, h0, c0, context_features,text_context, eos_threshold=0.5):
    """
    Generates a normalized stroke sequence using the LSTM, feeding its own
    predicted coordinate back as input (autoregressive decoding).
    """
    device = h0.device
    batch_size = h0.size(1)

    # Initialize output storage
    predicted_coords = []

    # LSTM states (h_t, c_t)
    hx, cx = h0.squeeze(0), c0.squeeze(0) # [B, H]

    # Initial input token is (0, 0)
    input_coord = torch.zeros(batch_size, 2).to(device) # [B, 2]

    # EOS status tracker
    is_eos = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for t in range(model.max_stroke_len):
        # 1. Attention: Get context vector based on current hidden state
        context_vector, _ = model.attention(context_features, hx) # [B, C]

        # 2. LSTM Input: Concatenate last predicted coordinate and context vector
        lstm_input = torch.cat([input_coord, context_vector,text_context], dim=1).unsqueeze(1) # [B, 1, 2+C+H]

        # 3. LSTM Step
        lstm_output, (hx_t, cx_t) = model.lstm(lstm_input, (hx.unsqueeze(0), cx.unsqueeze(0)))

        # Update states for next step
        hx, cx = hx_t.squeeze(0), cx_t.squeeze(0)

        # 4. Final Output: [B, 3] (x, y, EOS logit)
        output = model.output_layer(lstm_output.squeeze(1))

        next_coord_raw = output[:, :2] # Raw (x, y) prediction
        pred_xy = torch.tanh(next_coord_raw) # Bounded (x, y)
        # pred_xy = output[:, :2]
        pred_eos_logit = output[:, 2]

        # 5. Check EOS condition
        pred_eos_prob = torch.sigmoid(pred_eos_logit)
        newly_finished = (pred_eos_prob > eos_threshold) & (~is_eos)

        # Update EOS status: Once finished, keep it finished
        is_eos = is_eos | newly_finished

        # Apply a mask to prevent new coordinates from being added once EOS is reached
        if is_eos.all():
            break

        # Store predicted normalized coordinate for all steps (including steps after EOS)
        predicted_coords.append(pred_xy)

        # 6. Autoregressive step: The current prediction is the input for the next step
        # If a stroke is finished, the input for the next step should be fixed (e.g., 0)
        # to avoid generating junk, but we continue looping to fill max_len
        # or if other strokes in the batch are still generating.
        input_coord = pred_xy

    # Stack results: [B, N_steps, 2]
    if not predicted_coords:
        return torch.empty(batch_size, 0, 2).to(device)

    return torch.stack(predicted_coords, dim=1) # [B, L_actual, 2]

def run_inference_walkthrough(model :StrokeModel, img_x, labels):
    # Set model to evaluation mode
    model.eval()

    # --- STEP 1: Run Encoder and Parameter Head ---
    print("Step 1/4: Running Encoder and Parameter Head...")
    # This gets the transformation parameters and the initial LSTM state
    pred_params, h0, c0, context_features, text_context = model(img_x, labels)

    # --- STEP 2: Sequence Decoding ---
    print("Step 2/4: Running Autoregressive Sequence Decoder...")

    # The output is [B, L_actual, 2], containing the predicted normalized (x, y) coordinates
    # We set a max length and an EOS probability threshold
    pred_normed_coords_tensor = inference_decode_stroke(
        model,
        h0,
        c0,
        context_features,
        text_context,
        eos_threshold=0.7 # Often a high threshold is needed for clean breaks
    )

    print(f"   -> Predicted normalized sequence length: {pred_normed_coords_tensor.size(1)} points.")

    # Convert to numpy for utility function and un-normalization
    pred_params_np = pred_params.squeeze(0).cpu().detach().numpy()
    pred_normed_coords_np = pred_normed_coords_tensor.squeeze(0).cpu().numpy()


    # --- STEP 3: Preparing Data for Reconstruction (Numpy/Utility) ---
    print("Step 3/4: Preparing predicted data for reconstruction...")

    # A. Recreate the Padded Stroke for the Utility Function
    # The reconstruction function needs a full 'padded_stroke' array and an 'eos_arr'
    N_pred = pred_normed_coords_np.shape[0]

    # Pad to max_stroke_len with -1 (as in raw_stroke_to_normed)
    pad_len = model.max_stroke_len - N_pred
    padded_stroke_pred = np.pad(pred_normed_coords_np, ((0, pad_len), (0, 0)), constant_values=-1)

    # B. Create the EOS array (set EOS flag at the predicted length N_pred)
    eos_arr_pred = np.zeros(model.max_stroke_len, dtype=np.float32)
    if N_pred > 0:
        # Set EOS flag at the last predicted point index
        eos_arr_pred[N_pred - 1] = 1.0

    # --- STEP 4: Reconstruction (Using the utility function from the Canvas) ---
    print("Step 4/4: Reconstructing stroke geometry...")

    # Call the utility function with the predicted parameters and normalized coordinates
    reconstructed_stroke = transform_stroke(
        pred_params_np,
        padded_stroke_pred,
        eos_arr_pred
    )

    print(f"   -> Final reconstructed stroke shape (N x 2): {reconstructed_stroke.shape}")
    print("Inference complete. The reconstructed stroke is ready for drawing.")
    return reconstructed_stroke, pred_params_np,padded_stroke_pred,eos_arr_pred

# %% Test Inference
# force_label = 'apple'
force_label = 'cat'
test_idx = 55
for i,(img_x, labels, target_params, target_coords, target_eos) in enumerate(test_dataloader):
    if i>=test_idx:
        break
img_x = img_x.unsqueeze(1).to(model.device)
final_stroke, p, s, eos = run_inference_walkthrough(model,img_x[:1],labels[:1])
true_stroke = transform_stroke(target_params[0].cpu().detach().numpy(), target_coords[0].cpu().detach().numpy(), target_eos[0].cpu().detach().numpy())
p_img = img_x[0].cpu().detach().numpy()[0]
plt.imshow(p_img,cmap='gray');plt.plot(final_stroke[:,0],final_stroke[:,1]);plt.plot(true_stroke[:,0],true_stroke[:,1])
plt.title(f"Truth:{labels[0]}, Forced:{force_label} Testing:{test_idx}")
# %%
