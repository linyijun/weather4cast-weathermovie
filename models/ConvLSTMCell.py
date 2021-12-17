import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, bias=True):
        """
        params:
            in_channels (int) - number of channels in the input image
            h_channels (int) - number of channels of hidden state
            kernel_size (int, int) - size of the convolution kernel
            bias (bool, optional) - default: True
        """

        super(ConvLSTMCell, self).__init__()

        self.h_channels = h_channels
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, image_size):
        """ initialize the first hidden state as zeros """
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.h_channels, height, width, device=self.conv.weight.device))
