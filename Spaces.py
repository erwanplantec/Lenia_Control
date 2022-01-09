import torch
import numpy as np
from addict import Dict
import numbers

class Space(object):
    """
    Defines the parameter space(s) of a system - Inspired from OpenAI's gym API.
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    def sample(self):
        """
        Randomly sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def mutate(self, x):
        """
        Randomly mutate an element of this space.
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)


class DiscreteSpace(Space):
    """A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    /!\ mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example::

        >>> DiscreteSpace(2)

    """

    def __init__(self, n, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert n >= 0
        self.n = n

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)
        super(DiscreteSpace, self).__init__((), torch.int64)

    def sample(self):
        return torch.randint(self.n, ())

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std, ())
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape == ()):  # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, DiscreteSpace) and self.n == other.n



class BoxSpace(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> BoxSpace(low=-1.0, high=2.0, shape=(3, 4), dtype=torch.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> BoxSpace(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([2.0, 4.0]), dtype=torch.float32)
        Box(2,)

    """

    def __init__(self, low, high, shape=None, dtype=torch.float32, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match provided shape"
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not isinstance(low, numbers.Number):
            shape = low.shape
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not isinstance(high, numbers.Number):
            shape = high.shape
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if isinstance(low, numbers.Number):
            low = torch.full(shape, low, dtype=dtype)

        if isinstance(high, numbers.Number):
            high = torch.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low.type(self.dtype)
        self.high = high.type(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = ~torch.isneginf(self.low)
        self.bounded_above = ~torch.isposinf(self.high)

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(BoxSpace, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high.type(torch.float64) if self.dtype.is_floating_point else self.high.type(torch.int64) + 1
        sample = torch.empty(self.shape, dtype=torch.float64)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = torch.randn(unbounded[unbounded].shape, dtype=torch.float64)

        sample[low_bounded] = (-torch.rand(low_bounded[low_bounded].shape, dtype=torch.float64)).exponential_() + \
                              self.low[low_bounded]

        sample[upp_bounded] = self.high[upp_bounded] - (
            -torch.rand(upp_bounded[upp_bounded].shape, dtype=torch.float64)).exponential_()

        sample[bounded] = (self.low[bounded] - high[bounded]) * torch.rand(bounded[bounded].shape,
                                                                           dtype=torch.float64) + high[bounded]

        if not self.dtype.is_floating_point:  # integer
            sample = torch.floor(sample)

        return sample.type(self.dtype)

    def mutate(self, x, mask=None):
        if(mask==None):
          mask=torch.ones(x.shape).to(x.device)
        
        mutate_mask = mask*((torch.rand(self.shape) < self.indpb).type(torch.float64)).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        if not self.dtype.is_floating_point:  # integer
            x = torch.floor(x)
        x = x.type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        return x.shape == self.shape and torch.all(x >= torch.as_tensor(self.low, dtype=self.dtype, device=x.device)) and torch.all(x <= torch.as_tensor(self.high, dtype=self.dtype, device=x.device))

    def clamp(self, x):
        if self.is_bounded(manner="below"):
            x = torch.max(x, torch.as_tensor(self.low, dtype=self.dtype, device=x.device))
        if self.is_bounded(manner="above"):
            x = torch.min(x, torch.as_tensor(self.high, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "BoxSpace({}, {}, {}, {})".format(self.low.min(), self.high.max(), self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(other, BoxSpace) and (self.shape == other.shape) and torch.allclose(self.low,
                                                                                              other.low) and torch.allclose(
            self.high, other.high)


class DictSpace(Space):
    """
    A Dict dictionary of simpler spaces.

    Example usage:
    self.genome_space = spaces.DictSpace({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_genome_space = spaces.DictSpace({
        'sensors':  spaces.DictSpace({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.DictSpace({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.DictSpace({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs), 'Use either DictSpace(spaces=dict(...)) or DictSpace(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, list):
            spaces = Dict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, Space), 'Values of the attrdict should be instances of gym.Space'
        Space.__init__(self, None, None)  # None for shape and dtype, since it'll require special handling

    def sample(self):
        return Dict([(k, space.sample()) for k, space in self.spaces.items()])

    def mutate(self, x):
        return Dict([(k, space.mutate(x[k])) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def clamp(self, x):
        return Dict([(k, space.clamp(x[k])) for k, space in self.spaces.items()])

    def __getitem__(self, key):
        return self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return "DictSpace(" + ", ".join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def __eq__(self, other):
        return isinstance(other, DictSpace) and self.spaces == other.spaces



class MultiDiscreteSpace(Space):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):

        """
        nvec: vector of counts of each categorical variable
        """
        assert (torch.tensor(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = torch.as_tensor(nvec, dtype=torch.int64)
        self.mutation_std = mutation_std

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.nvec.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.nvec.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.nvec.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiDiscreteSpace, self).__init__(self.nvec.shape, torch.int64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
        mutate_mask = (torch.rand(self.shape) < self.indpb).to(x.device)
        noise = torch.normal(self.mutation_mean, self.mutation_std).to(x.device)
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(self.nvec == other.nvec)



class BoxGoalSpace(BoxSpace):
    def __init__(self, representation, autoexpand=True, low=0., high=0., shape=None, dtype=torch.float32):
        self.representation = representation
        self.autoexpand = autoexpand
        if shape is not None:
            if isinstance(shape, list) or isinstance(shape, tuple):
                assert len(shape) == 1 and shape[0] == self.representation.n_latents
            elif isinstance(shape, numbers.Number):
                assert shape == self.representation.n_latents
        BoxSpace.__init__(self, low=low, high=high, shape=(self.representation.n_latents,), dtype=dtype)

    def map(self, observations, **kwargs):
        embedding = self.representation.calc(observations, **kwargs)
        if self.autoexpand:
            embedding_c = embedding.detach()
            is_nan_mask = torch.isnan(embedding_c)
            if is_nan_mask.sum() > 0:
                embedding_c[is_nan_mask] = self.low[is_nan_mask]
                self.low = torch.min(self.low, embedding_c)
                embedding_c[is_nan_mask] = self.high[is_nan_mask]
                self.high = torch.max(self.high, embedding_c)
            else:
                self.low = torch.min(self.low, embedding_c)
                self.high = torch.max(self.high, embedding_c)
        return embedding

    def calc_distance(self, embedding_a, embedding_b, **kwargs):
        return self.representation.calc_distance(embedding_a, embedding_b, **kwargs)

    def sample(self):
        return BoxSpace.sample(self)

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

