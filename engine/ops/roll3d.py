"""Roll 3D."""

from mindspore import nn
from mindspore import ops

__all__ = ['Roll3D']


class Roll3D(nn.Cell):
    """
    Roll Tensors of shape (B, D, H, W, C).
    TODO: Compare to torch.roll there is a dim left. and where is the dim?

    Args:
        shift (tuple[int]): shift size for target rolling.

    Inputs:
        Tensor of shape (B, D, H, W, C).

    Outputs:
        Rolled Tensor.
    """

    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        self.concat_1 = ops.Concat(axis=1)
        self.concat_2 = ops.Concat(axis=2)
        self.concat_3 = ops.Concat(axis=3)

    def construct(self, x):
        """Construct a Roll3D ops."""
        x = self.concat_1(
            (x[:, -self.shift[0]:, :, :],
             x[:, :-self.shift[0], :, :]))
        x = self.concat_2(
            (x[:, :, -self.shift[1]:, :],
             x[:, :, :-self.shift[1], :]))
        x = self.concat_3(
            (x[:, :, :, -self.shift[2]:],
             x[:, :, :, :-self.shift[2]]))
        return x
