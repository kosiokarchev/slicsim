from fractions import Fraction

from phytorch.units.unit import Dimension, Unit


ADU = Unit({Dimension('ADU'): 1}, value=1., name='ADU')
electrons = Unit(value=1., name='e-')

px = Unit({Dimension('px'): 1}, value=1., name='px')
linpx = px ** Fraction(1, 2)
