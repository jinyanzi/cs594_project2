require 'nn'
-- require 'cudnn'
-- require 'cunn'
-- local nninit = require 'nninit'
local Convolution = nn.SpatialConvolution
local SBatchNorm = nn.SpatialBatchNormalization
local ReLU = nn.ReLU
local ResidualBlock, Parent = torch.class('nn.ResidualBlock', 'nn.Container')

function ResidualBlock:__init(nInputPlane, nOutputPlane, stride, p)

	Parent.__init(self)
	self.p = p or 0
	self.train = true
	self.gate = true
	-- The new Residual Unit in [a]
	local nBottleneckPlane = nOutputPlane / 4
	-- if opt.resnet_nobottleneck then
	--   nBottleneckPlane = nOutputPlane
	-- end
	
	if nInputPlane == nOutputPlane then -- most Residual Units have this shape      
		self.net = nn.Sequential()

		 -- conv1x1
		self.net:add(SBatchNorm(nInputPlane))
		self.net:add(ReLU(true))
		self.net:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
		
		-- conv3x3
		self.net:add(SBatchNorm(nBottleneckPlane))
		self.net:add(ReLU(true))
		self.net:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
		
		-- conv1x1
		self.net:add(SBatchNorm(nBottleneckPlane))
		self.net:add(ReLU(true))
		self.net:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
		
		self.skip = nn.Identity()
		self.skip:add(nn.Identity())

		--self.modules = nn.Sequential()
		--   :add(nn.ConcatTable()
		--   :add(self.net)
		--   :add(self.skip))
		--   :add(nn.CAddTable(true))
		self.modules = {self.net, self.skip}

	else -- Residual Units for increasing dimensions
		self.net= nn.Sequential()
		
		-- common BN, ReLU
		self.net:add(SBatchNorm(nInputPlane))
		self.net:add(ReLU(true))
		
		-- conv1x1
		self.net:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
		
		-- conv3x3
		self.net:add(SBatchNorm(nBottleneckPlane))
		self.net:add(ReLU(true))
		self.net:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
		
		-- conv1x1
		self.net:add(SBatchNorm(nBottleneckPlane))
		self.net:add(ReLU(true))
		self.net:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
		
		self.skip = nn.Sequential()
		self.skip:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
		
		--self.modules = nn.Sequential()
		--   :add(nn.ConcatTable()
		--   :add(self.net)
		--   :add(self.skip))
		--   :add(nn.CAddTable(true))
		self.modules = {self.net, self.skip}
	end
end


--function ResidualBlock:__init(deathRate, nInputPlane, nOutputPlane, stride)
--    parent.__init(self)
--    self.gradInput = torch.Tensor()
--    self.gate = true
--    self.train = true
--    self.deathRate = deathRate
--    nOutputPlane = nOutputPlane or nInputPlane
--    stride = stride or 1
--
--    self.net = nn.Sequential()
--    self.net:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, stride,stride, 1,1)
--                                             :init('weight', nninit.kaiming, {gain = 'relu'})
--                                             :init('bias', nninit.constant, 0))
--    self.net:add(cudnn.SpatialBatchNormalization(nOutputPlane))
--    self.net:add(cudnn.ReLU(true))
--    self.net:add(cudnn.SpatialConvolution(nOutputPlane, nOutputPlane,
--                                      3,3, 1,1, 1,1)
--                                      :init('weight', nninit.kaiming, {gain = 'relu'})
--                                      :init('bias', nninit.constant, 0))
--    self.net:add(cudnn.SpatialBatchNormalization(nOutputPlane))
--    self.skip = nn.Sequential()
--    self.skip:add(nn.Identity())
--    if stride > 1 then
--       -- optional downsampling
--       self.skip:add(nn.SpatialAveragePooling(1, 1, stride,stride))
--    end
--    if nOutputPlane > nInputPlane then
--       -- optional padding, this is option A in their paper
--       self.skip:add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
--    elseif nOutputPlane < nInputPlane then
--       print('Do not do this! nOutputPlane < nInputPlane!')
--    end
--    
--    self.modules = {self.net, self.skip}
--end

function ResidualBlock:updateOutput(input)
    local skip_forward = self.skip:forward(input)
    self.output:resizeAs(skip_forward):copy(skip_forward)
    if self.train then
      if self.gate then -- only compute convolutional output when gate is open
        self.output:add(self.net:forward(input))
      end
    else
      self.output:add(self.net:forward(input):mul(1-self.p))
    end
    return self.output
end

function ResidualBlock:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.gate then
      self.gradInput:add(self.net:updateGradInput(input, gradOutput))
   end
   return self.gradInput
end

function ResidualBlock:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gate then
      self.net:accGradParameters(input, gradOutput, scale)
   end
end

---- Adds a residual block to the passed in model ----
-- function addResidualBlock(model, deathRate, nInputPlane, nOutputPlane, stride)
--    model:add(nn.ResidualBlock(deathRate, nInputPlane, nOutputPlane, stride))
--    model:add(cudnn.ReLU(true))
--    return model
-- end
