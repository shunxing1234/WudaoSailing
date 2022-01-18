# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.cuda.is_available():
    from .base_model import BaseModel
    from .cached_autoregressive_model import CachedAutoregressiveModel
    from .cuda2d_model import Cuda2dModel
    from .glm_model import GLMModel
    from .encoder_decoder_model import EncoderDecoderModel
    from .distributed import PyTorchDistributedDataParallel, DistributedDataParallel
