
import torch
from Spaces import *

""" =============================================================================================
Lenia Update Rule Space: 
============================================================================================= """


class LeniaUpdateRuleSpace(DictSpace):

    @staticmethod
    def default_config():
        default_config = Dict()
        return default_config

    def __init__(self,nb_k=10,C=1, config={}, **kwargs):
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.config.update(kwargs)
        print(C)
        spaces = Dict(
            R = DiscreteSpace(n=25, mutation_mean=0.0, mutation_std=0.01, indpb=0.01),
            c0= MultiDiscreteSpace(nvec=[C]*nb_k, mutation_mean=torch.zeros((nb_k,)), mutation_std=0.1*torch.ones((nb_k,)), indpb=0.1),
            c1= MultiDiscreteSpace(nvec=[C]*nb_k, mutation_mean=torch.zeros((nb_k,)), mutation_std=0.1*torch.ones((nb_k,)), indpb=0.1),
            T = BoxSpace(low=1.0, high=10.0, shape=(), mutation_mean=0.0, mutation_std=0.1, indpb=0.01, dtype=torch.float32),
            rk = BoxSpace(low=0, high=1, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            b = BoxSpace(low=0.0, high=1.0, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            w = BoxSpace(low=0.01, high=0.5, shape=(nb_k,3), mutation_mean=torch.zeros((nb_k,3)), mutation_std=0.2*torch.ones((nb_k,3)), indpb=1, dtype=torch.float32),
            m = BoxSpace(low=0.05, high=0.5, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32),
            s = BoxSpace(low=0.001, high=0.18, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.01**torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32),
            h = BoxSpace(low=0, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=0.1, dtype=torch.float32),
            r = BoxSpace(low=0.2, high=1.0, shape=(nb_k,), mutation_mean=torch.zeros((nb_k,)), mutation_std=0.2*torch.ones((nb_k,)), indpb=1, dtype=torch.float32)
            #kn = DiscreteSpace(n=4, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
            #gn = DiscreteSpace(n=3, mutation_mean=0.0, mutation_std=0.1, indpb=1.0),
        )

        DictSpace.__init__(self, spaces=spaces)
    def mutate(self,x):
      mask=(x['s']>0.04).float()*(torch.rand(x['s'].shape[0])<0.25).float().to(x['s'].device)
      param=[]
      for k, space in self.spaces.items():
        if(k=="R" or k=="c0" or k=="c1" or k=="T"):
          param.append((k, space.mutate(x[k])))
        elif(k=='rk' or k=='w' or k=='b'):
          param.append((k, space.mutate(x[k],mask.unsqueeze(-1))))
        else:
          param.append((k, space.mutate(x[k],mask)))

      return Dict(param)


""" =============================================================================================
Lenia Main
============================================================================================= """

bell = lambda x, m, s: torch.exp(-((x-m)/s)**2 / 2) 
# Lenia family of functions for the kernel K and for the growth mapping g
kernel_core = {
    0: lambda u: (4 * u * (1 - u)) ** 4,  # polynomial (quad4)
    1: lambda u: torch.exp(4 - 1 / (u * (1 - u))),  # exponential / gaussian bump (bump4)
    2: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float(),  # step (stpz1/4)
    3: lambda u, q=1 / 4: (u >= q).float() * (u <= 1 - q).float() + (u < q).float() * 0.5,  # staircase (life)
    4: lambda u: torch.exp(-(u-0.5)**2/0.2),
    8: lambda u: (torch.sin(10*u)+1)/2,
    9: lambda u: (a*torch.sin((u.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2

}
field_func = {
    0: lambda n, m, s: torch.max(torch.zeros_like(n), 1 - (n - m) ** 2 / (9 * s ** 2)) ** 4 * 2 - 1, # polynomial (quad4)
    1: lambda n, m, s: torch.exp(- (n - m) ** 2 / (2 * s ** 2)-1e-3) * 2 - 1,  # exponential / gaussian (gaus)
    2: lambda n, m, s: (torch.abs(n - m) <= s).float() * 2 - 1 , # step (stpz)
}

# ker_c =lambda r,a,b,c :(a*torch.sin((r.unsqueeze(-1)*5*b+c)*np.pi)).sum(-1)/(2*a.sum())+1/2
ker_c= lambda x,r,w,b : (b*torch.exp(-((x.unsqueeze(-1)-r)/w)**2 / 2)).sum(-1) 


# Lenia Step FFT version (faster)
class LeniaStepFFTC(torch.nn.Module):
    """ Module pytorch that computes one Lenia Step with the fft version"""

    def __init__(self,C, R, T,c0,c1,r,rk, b,w,h, m, s, gn, is_soft_clip=False, SX=256, SY=256, device='cuda'):
        torch.nn.Module.__init__(self)

        self.register_buffer('R', R)
        self.register_buffer('T', T)
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)
        self.register_parameter('r', torch.nn.Parameter(r))
        self.register_parameter('rk', torch.nn.Parameter(rk))
        self.register_parameter('b', torch.nn.Parameter(b))
        self.register_parameter('w', torch.nn.Parameter(w))
        self.register_parameter('h', torch.nn.Parameter(h))
        self.register_parameter('m', torch.nn.Parameter(m))
        self.register_parameter('s', torch.nn.Parameter(s))

        self.gn = 1
        self.nb_k=c0.shape[0]

        self.SX = SX
        self.SY = SY
        
        self.is_soft_clip = is_soft_clip
        self.C=C

        self.device = device
        self.to(device)
        self.kernels=torch.zeros((self.nb_k,self.SX,self.SY,2)).to(self.device)

        self.compute_kernel()


    def compute_kernel(self):
      
      x = torch.arange(self.SX).to(self.device)
      y = torch.arange(self.SY).to(self.device)
      xx = x.view(-1, 1).repeat(1, self.SY)
      yy = y.repeat(self.SX, 1)
      X = (xx - int(self.SX / 2)).float() 
      Y = (yy - int(self.SY / 2)).float() 
      self.kernels=torch.zeros((self.nb_k,self.SX,self.SY,2)).to(self.device)
      for i in range(self.nb_k):

        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)/ ((self.R+15)*self.r[i])
        
        kernel = torch.sigmoid(-(D-1)*10) * ker_c(D,self.rk[i],self.w[i],self.b[i])
        kernel_sum = torch.sum(kernel)
        # normalization of the kernel
        kernel_norm = (kernel / kernel_sum).unsqueeze(0).unsqueeze(0)
        # plt.imshow(kernel_norm[0,0].detach().cpu()*100)
        # plt.show()
        # fft of the kernel
        kernel_FFT = torch.rfft(kernel_norm, signal_ndim=2, onesided=False).to(self.device)
        self.kernels[i]=kernel_FFT

    def forward(self, input):
        
        self.D=torch.zeros(input.shape).to(self.device)
        self.Dn=torch.zeros(self.C)
        
        world_FFT = [torch.rfft(input[:,:,:,i], signal_ndim=2, onesided=False) for i in range(self.C)]
        
        ## speed up for 1 channel creature vectorizing
        if(self.C==1):
          world_FFT_c = world_FFT[0]
          potential_FFT = complex_mult_torch(self.kernels, world_FFT_c)
          potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
          potential = roll_n(potential, 2, potential.size(2) // 2)
          potential = roll_n(potential, 1, potential.size(1) // 2)
          gfunc = field_func[min(self.gn, 3)]
          field = gfunc(potential, self.m.unsqueeze(-1).unsqueeze(-1), self.s.unsqueeze(-1).unsqueeze(-1))
          self.D[:,:,:,0]=(self.h.unsqueeze(-1).unsqueeze(-1)*field).sum(0,keepdim=True)
          self.Dn[0]=self.h.sum()


        ##Base version for multi channel
        else:
          for i in range(self.nb_k):
            c0b=int((self.c0[i]))
            c1b=int((self.c1[i]))
            
            world_FFT_c = world_FFT[c0b]
            potential_FFT = complex_mult_torch(self.kernels[i].unsqueeze(0), world_FFT_c)
            
            potential = torch.irfft(potential_FFT, signal_ndim=2, onesided=False)
            potential = roll_n(potential, 2, potential.size(2) // 2)
            potential = roll_n(potential, 1, potential.size(1) // 2)


            gfunc = field_func[min(self.gn, 3)]
            field = gfunc(potential, self.m[i], self.s[i])

            self.D[:,:,:,c1b]=self.D[:,:,:,c1b]+self.h[i]*field
            self.Dn[c1b]=self.Dn[c1b]+self.h[i]


        
        
        if not self.is_soft_clip:
            # output_img = torch.sigmoid((input + (1.0 / self.T) * self.D-0.5)*10)
            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)
            # output_img = torch.tanh(input + (1.0 / self.T) * self.D)
        else:
            # output_img = soft_clip(input + (1.0 / self.T) * self.D, 0, 1, self.T)
            # output_img = input + (1.0 / self.T) * ((self.D/self.Dn+1)/2-input)
            output_img = torch.clamp(input + (1.0 / self.T) * self.D, min=0., max=1.)

        return output_img





class Lenia_C( torch.nn.Module):

    @staticmethod
    def default_config():
        default_config = Dict()
        default_config.version = 'pytorch_fft'  # "pytorch_fft", "pytorch_conv2d"
        default_config.SX = 256
        default_config.SY = 256
        default_config.final_step = 40
        default_config.C = 1

        return default_config


    def __init__(self,nb_k=10,C=1, initialization_space=None, update_rule_space=None,  config={}, device=torch.device('cpu'), **kwargs):
        torch.nn.Module.__init__(self)
        self.config = self.__class__.default_config()
        self.config.update(config)
        self.device = device
        if update_rule_space is not None:
            self.update_rule_space = update_rule_space
        else:
            self.update_rule_space = LeniaUpdateRuleSpace(nb_k,C)
        
        self.config.C=C
        self.init=torch.rand(1,60,60,self.config.C)
        self.init[:20,:20]=torch.clip(self.init[:20,:20]*2,0,1)
        self.run_idx = 0
        
        self.reset()
        
        



    def reset(self,  update_rule_parameters=None):
        # call the property setters
        if(update_rule_parameters is not None):
          self.update_rule_parameters = update_rule_parameters
        else:
          policy_parameters = Dict.fromkeys(['update_rule'])
          policy_parameters['update_rule'] = self.update_rule_space.sample()
          #divide h by 3 at the beginning as some unbalanced kernels can easily kill
          policy_parameters['update_rule'].h =policy_parameters['update_rule'].h/3
          self.update_rule_parameters = policy_parameters['update_rule']
        # initialize Lenia CA with update rule parameters
        if self.config.version == "pytorch_fft":
            lenia_step = LeniaStepFFTC(self.config.C,self.update_rule_parameters['R'], self.update_rule_parameters['T'],self.update_rule_parameters['c0'],self.update_rule_parameters['c1'], self.update_rule_parameters['r'], self.update_rule_parameters['rk'], self.update_rule_parameters['b'], self.update_rule_parameters['w'],self.update_rule_parameters['h'], self.update_rule_parameters['m'],self.update_rule_parameters['s'],1, is_soft_clip=False, SX=self.config.SX, SY=self.config.SY, device=self.device)
        self.add_module('lenia_step', lenia_step)        
        # push the nn.Module and the available devoce
        self.to(self.device)
        self.generate_init_state()



        

    def generate_init_state(self):
        init_state = torch.zeros( 1,self.config.SX, self.config.SY,self.config.C, dtype=torch.float64)
        # init_state[0,0, 180:180+cppn_output_height, 180:180+cppn_output_width,:self.config.C] = cppn_net_output.unsqueeze(-1)
        init_state[0,98:158,self.config.SY-158:self.config.SY-98]=self.init
        self.state = init_state.to(self.device)
        self.step_idx = 0

    def step(self, intervention_parameters=None):
 
        self.state = self.lenia_step(self.state)
        self.step_idx += 1

        return self.state


    def forward(self):
        state = self.step(None)
        return state

    def run(self): 
        # self.generate_init_state()
        observations = Dict()
        observations.timepoints = list(range(self.config.final_step))
        observations.states = torch.empty((self.config.final_step, self.config.SX, self.config.SY,self.config.C))
        observations.states[0]  = self.state
 
        for step_idx in range(1, self.config.final_step):
            cur_observation = self.step(None)
            observations.states[step_idx] = cur_observation[0,:,:,:]


        return observations



    def update_update_rule_parameters(self):
        new_update_rule_parameters = deepcopy(self.update_rule_parameters)
        new_update_rule_parameters['m'] = self.lenia_step.m.data
        new_update_rule_parameters['s'] = self.lenia_step.s.data
        new_update_rule_parameters['r'] = self.lenia_step.r.data
        new_update_rule_parameters['rk'] = self.lenia_step.rk.data
        new_update_rule_parameters['b'] = self.lenia_step.b.data
        new_update_rule_parameters['w'] = self.lenia_step.w.data
        new_update_rule_parameters['h'] = self.lenia_step.h.data
        
        if not self.update_rule_space.contains(new_update_rule_parameters):
            new_update_rule_parameters = self.update_rule_space.clamp(new_update_rule_parameters)
            warnings.warn('provided parameters are not in the space range and are therefore clamped')
        self.update_rule_parameters = new_update_rule_parameters

    


    