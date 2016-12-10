--  ResNet-1001
--  This is a re-implementation of the 1001-layer residual networks described in:
--  [a] "Identity Mappings in Deep Residual Networks", arXiv:1603.05027, 2016,
--  authored by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

--  Acknowledgement: This code is contributed by Xiang Ming from Xi'an Jiaotong Univeristy.

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
local utils = paths.dofile'utils.lua'
local residual = paths.dofile'residualBlock.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	
   local depth = opt.depth
   
   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride, p)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride, p))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1, p))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
      local n = (depth - 2) / 9

      -- The new ResNet-164 and ResNet-1001 in [a]
      local nStages = {16, 64, 128, 256}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(nn.ResidualBlock, nStages[1], nStages[2], n, 1, opt.stoDrop)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(nn.ResidualBlock, nStages[2], nStages[3], n, 2, opt.stoDrop)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(nn.ResidualBlock, nStages[3], nStages[4], n, 2, opt.stoDrop)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   -- model:get(1).gradInput = nil

   return model
end

return createModel
