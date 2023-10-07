# S-DeepONet
A sequential DeepONet model implementation that uses a recurrent neural network (GRU and LSTM) in the branch and a feed-forward neural network in the trunk. The branch network efficiently encodes time-dependent input functions, and the trunk network captures the spatial dependence of the full-field data.

The DeepONet implementation and training is based on DeepXDE:
@article{lu2021deepxde,
  title={DeepXDE: A deep learning library for solving differential equations},
  author={Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  journal={SIAM review},
  volume={63},
  number={1},
  pages={208--228},
  year={2021},
  publisher={SIAM}
}

If you find our model helpful in your specific applications and researches, please cite this article as: 
To be updated.


The training data is large in size and can be downloaded through the following UIUC Box link:
https://uofi.box.com/s/5uofjmix7e9fr7muojgrb0zd264pu922