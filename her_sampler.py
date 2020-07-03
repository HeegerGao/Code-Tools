'''
    her_sampler from openai baseline
'''
import numpy as np

class HerSampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        '''
        replay_strategy:一般reply_strategy使用future，表示从sample样本之后的状态中找ag
        reply_k:表示结果的样本中被替换g的样本数量是没有被替换g的样本数量的几倍。如reply_k=4,表示
                在sample一个batch_size = 10的结果中，有8个是被替换g的样本，有2个是没有被替换g的样本。
                reply_k越大，表示结果中使用her样本的比例越高。但是注意reply_k是概率选的，也就是说
                不是绝对k倍，只是统计意义上的k倍。
        '''
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        '''
        episode_batch:  整个buffer,或者说准备被sample的buffer数据。其结构必须为：
                        episode 0: -----------------
                        episode 1: -----------------
                        ...
                        ...
                        ...
                        episode n: -----------------
                                   |<------T------>|
                        其中，每个-表示一个transition，是一个字典，即：
                            - = 
                            {
                                'obs': ……,
                                'g': ……,
                                'ag': ……,
                                'obs_next': ……,
                                'ag_next': ……,
                                'actions': ……,
                                'r': ……
                            }
        batch_size_in_transitions: 这次sample出来的transition的数量，注意是
                                    transition的数量，所以是n个episode里面
                                    选了batch_size个transition，最后结果就类似这样：
                        episode 0: ---*-------------
                        episode 1: ------*----------
                        ...
                        ...        -------------*---
                        ...
                        episode n: --*--------------
                                   |<------T------>|
                        其中*的个数加起来等于batch_size
        '''

        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        # episode_idxs是个一维数组，长度为batch_size，是中间变量，表示如果
        # 给的buffer的episode比batch_size还多，则选batch_size个episode，
        # 如果少，则重复选一些，总共batch_size条episode
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples也是中间变量，一维数组，长度为batch_size，和episode_idxs一起使用，
        # 也就是我们现在只在一个batch_size*batch_size的方块中sample了，其他不管了
        t_samples = np.random.randint(T, size=batch_size)
        # 注意[episode_idxs, t_samples]这样写，最后transitions的结果是个一维数组，
        # 而不是二维数组。结果是[episode_idxs[i],t_samples[i] for i in range(batch_size)]
        # 另外注意要copy！！！！！！！！！！！！！！！！！！！！！因为后面要替换g
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        #her_indexes是batchsize里面按照0.8的概率sample，一个一维数组
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        # future_offset表示刚刚得到的batch_size个transitions，各自对应的在T中的t 加上一个到T末尾的随机值，
        # 这种(T - t_samples)的写法很巧妙，把每个t都对应了一个future的新t，而且是随机的
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # 表示从transition中0.8的概率选出来一些，然后这些样本找一下各自轨迹中对应的一个future样本，
        # 也就是说我们现在把batch_size * 0.8 数量的样本都准备好替换g了
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
