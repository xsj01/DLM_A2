from nfd.dataset.push_dataset import *
# from vgpl.data import find_relations_neighbor, find_k_relations_neighbor, normalize_scene_param, denormalize
from nfd.utils.utils import get_object_poses

from torch.utils.data.dataloader import default_collate

# object_keys = [str(i) for i in range()]
# block_size = (0.01, 0.01, 0.01)

# n_particle_object=50
n_particle_pusher=10


# # DPI functions
# def prepare_input(positions, n_particle, n_shape, args, var=False):
#     '''
#     Input:
#         positions: (n_p + n_s) x 7 pose
#         n_particle: number of particles
#         n_shape: number of pusher
#     Output: [attr, particle, Rr, Rs]
#         attr: (n_p + n_s) x attr_dim
#         particle (unnormalized): (n_p + n_s) x state_dim
#         Rr, Rs: n_rel x (n_p + n_s)
#     '''
#     # positions: (n_p + n_s) x 3

#     verbose = args.verbose_data

#     count_nodes = n_particle + n_shape

#     if verbose:
#         print("prepare_input::positions", positions.shape)
#         print("prepare_input::n_particle", n_particle)
#         print("prepare_input::n_shape", n_shape)

#     ### object attributes
#     attr = np.zeros((count_nodes, args.attr_dim))

#     ##### add env specific graph components
#     # import pdb; pdb.set_trace()
#     rels = []
#     if args.env == 'PilePush':
#         attr[:n_particle-n_particle_pusher, 0] = 1
#         attr[n_particle-n_particle_pusher:-1, 1] = 1
#         pos = positions.data.cpu().numpy() if var else positions
#         # dis = np.sqrt(np.sum((pos[n_particle,-3:] - pos[:n_particle,-3:])**2, 1))
#         # nodes = np.nonzero(dis < args.neighbor_radius+0.1)[0]

#         # '''
#         # if verbose:
#         #     visualize_neighbors(pos, pos, 0, nodes)
#         #     print(np.sort(dis)[:10])
#         # '''

#         # pin = np.ones(nodes.shape[0], dtype=int) * n_particle
#         # rels += [np.stack([nodes, pin], axis=1)]
#     else:
#         AssertionError("Unsupported env %s" % str(args.env))
    
#     # if args.env in ['RigidFall', 'MassRope']:
#     queries = np.arange(n_particle)
#     anchors = np.arange(n_particle)

#     ##### add relations between leaf particles

#     rels += find_relations_neighbor(pos[:,-3:], queries, anchors, args.neighbor_radius, 2, var)
#     # rels += find_k_relations_neighbor(args.neighbor_k, pos, queries, anchors, args.neighbor_radius, 2, var)
#     # print(rels)
#     # print(pos[:,-3:])
#     if len(rels) > 0:
#         rels = np.concatenate(rels, 0)

#     if verbose:
#         print("Relations neighbor", rels.shape)

#     n_rel = rels.shape[0]
#     Rr = torch.zeros(n_rel, n_particle + n_shape)
#     Rs = torch.zeros(n_rel, n_particle + n_shape)
#     Rr[np.arange(n_rel), rels[:, 0]] = 1
#     Rs[np.arange(n_rel), rels[:, 1]] = 1

#     if verbose:
#         print("Object attr:", np.sum(attr, axis=0))
#         print("Particle attr:", np.sum(attr[:n_particle], axis=0))
#         print("Shape attr:", np.sum(attr[n_particle:n_particle + n_shape], axis=0))

#     if verbose:
#         print("Particle positions stats")
#         print("  Shape", positions.shape)
#         print("  Min", np.min(positions[:n_particle], 0))
#         print("  Max", np.max(positions[:n_particle], 0))
#         print("  Mean", np.mean(positions[:n_particle], 0))
#         print("  Std", np.std(positions[:n_particle], 0))

#     if var:
#         particle = positions
#     else:
#         particle = torch.FloatTensor(positions)

#     if verbose:
#         for i in range(count_nodes - 1):
#             if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
#                 print(i, attr[i], attr[i + 1])

#     attr = torch.FloatTensor(attr)
#     assert attr.size(0) == count_nodes
#     assert attr.size(1) == args.attr_dim

#     # attr: (n_p + n_s) x attr_dim
#     # particle (unnormalized): (n_p + n_s) x state_dim
#     # Rr, Rs: n_rel x (n_p + n_s)
#     return attr, particle, Rr, Rs

# def get_env_group(args, n_particles, scene_params):
#     use_gpu = args.use_gpu
#     # n_particles (int)
#     # scene_params: B x param_dim
#     # import pdb; pdb.set_trace()
#     B = scene_params.shape[0]
#     n_particle_object = n_particles-args.n_particle_pusher
#     p_rigid = torch.zeros(B, args.n_instance)
#     p_instance = torch.zeros(B, n_particles, args.n_instance)
#     physics_param = torch.zeros(B, n_particles)

#     if args.env == 'PilePush':
#         # norm_size = 0.1#normalize_scene_param(scene_params, 0, args.physics_param_range)
#         physics_param[:] = torch.FloatTensor(scene_params).view(B, 1)

#         # p_rigid[:, 0] = 1
#         # p_instance[:, :, 0] = 1
#         p_rigid[:, 1] = 1
#         p_instance[:, n_particle_object:, 1] = 1
#         p_instance[:, :n_particle_object, 0] = 1


#     # elif args.env == 'RigidFall':
#     #     norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
#     #     physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)

#     #     p_rigid[:] = 1

#     #     for i in range(args.n_instance):
#     #         p_instance[:, 64 * i:64 * (i + 1), i] = 1

#     # elif args.env == 'MassRope':
#     #     norm_stiff = normalize_scene_param(scene_params, 4, args.physics_param_range)
#     #     physics_param[:] = torch.FloatTensor(norm_stiff).view(B, 1)

#     #     n_rigid_particle = 81

#     #     p_rigid[:, 0] = 1
#     #     p_instance[:, :n_rigid_particle, 0] = 1
#     #     p_instance[:, n_rigid_particle:, 1] = 1

#     else:
#         raise AssertionError("Unsupported env")

#     if use_gpu:
#         p_rigid = p_rigid.cuda()
#         p_instance = p_instance.cuda()
#         physics_param = physics_param.cuda()

#     # p_rigid: B x n_instance
#     # p_instance: B x n_p x n_instance
#     # physics_param: B x n_p
#     return [p_rigid, p_instance, physics_param]

# def convert2dpi(object_poses, pusher_poses, args):
#     '''
#     Input: 
#         object_poses: TxNx7 
#         pusher_poses: Tx7
#     Output: attr, particles, n_particle, n_shape, scene_params, Rr, Rs
#     '''
#     T = object_poses.shape[0]
    
#     attrs, particles, Rrs, Rss = [], [], [], []
#     max_n_rel = 0

#     n_particle = object_poses.shape[1]
#     n_shape = 1
#     block_size = 0.1

#     scene_params = np.array([block_size])

#     poses = np.concatenate([object_poses, pusher_poses[:,None, :]], 1)

#     for t in range(T):
#         # positions = poses[t,:,:3]#.detach().cpu().numpy()
#         attr, particle, Rr, Rs = prepare_input(poses[t,...], n_particle, n_shape, args)

#         max_n_rel = max(max_n_rel, Rr.size(0))

#         attrs.append(attr)
#         particles.append(particle.numpy())
#         Rrs.append(Rr)
#         Rss.append(Rs)
    
#     # attr: (n_p + n_s) x attr_dim
#     # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
#     # scene_params: param_dim
#     attr = torch.FloatTensor(attrs[0])
#     particles = torch.FloatTensor(np.stack(particles))
#     scene_params = torch.FloatTensor(scene_params)

#     # pad the relation set
#     # Rr, Rs: seq_length x n_rel x (n_p + n_s)
#     if args.stage in ['dy']:
#         for i in range(len(Rrs)):
#             Rr, Rs = Rrs[i], Rss[i]
#             Rr = torch.cat([Rr, torch.zeros(max_n_rel - Rr.size(0), n_particle + n_shape)], 0)
#             Rs = torch.cat([Rs, torch.zeros(max_n_rel - Rs.size(0), n_particle + n_shape)], 0)
#             Rrs[i], Rss[i] = Rr, Rs
#         Rr = torch.FloatTensor(np.stack(Rrs))
#         Rs = torch.FloatTensor(np.stack(Rss))

#     return attr, particles, n_particle, n_shape, scene_params, Rr, Rs

def my_collate_all(batch):
    # print(batch[0]['pose'].shape)
    keys = list(batch[0].keys())
    keys.pop(keys.index('dpi'))
    # print(keys)
    output = default_collate([{key: torch.FloatTensor(d[key]) for key in keys} for d in batch])
    for key in ['dpi']:

        len_batch = len(batch[0]['dpi'])
        len_rel = 2

        ret = []
        for i in range(len_batch - len_rel):
            d = [item[key][i] for item in batch]
            if isinstance(d[0], int):
                d = torch.LongTensor(d)
            else:
                d = torch.FloatTensor(torch.stack(d))
            ret.append(d)

        # processing relations
        # R: B x seq_length x n_rel x (n_p + n_s)
        for i in range(len_rel):
            R = [item[key][-len_rel + i] for item in batch]
            max_n_rel = 0
            seq_length, _, N = R[0].size()
            for j in range(len(R)):
                max_n_rel = max(max_n_rel, R[j].size(1))
            for j in range(len(R)):
                r = R[j]
                r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
                R[j] = r

            R = torch.FloatTensor(torch.stack(R))

            ret.append(R)
        output[key] = tuple(ret)

    return output

def dpi_collate(batch):
    # print(batch)
    len_batch = len(batch[0])
    len_rel = 2

    ret = []
    for i in range(len_batch - len_rel):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i] for item in batch]
        max_n_rel = 0
        _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(0))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(max_n_rel - r.size(0), N)], 0)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    return tuple(ret)

class DynamicsDataset(PushDataset):
    def __init__(self, *args, **kwargs):
        self.cfg = kwargs.pop('cfg', None)
        super(DynamicsDataset, self).__init__(*args, **kwargs)
        

    def __getitem__(self, idx):
        if self.overfit:
            idx = idx%len(self.overfit)
            idx = self.overfit[idx]
            # print('overfit', idx)
        transform = self.transform
        if self.cache and self.data_list[idx]:#and len(self.data_list)==len(self):
            data = self.data_list[idx]
            # states = data['states']
            # actions = data['action_img']
            object_poses = data['object_poses']
            poses = get_pose(data['action'])
        else:
            data = load_data(self.file_list[idx])
            # color = torch.tensor(np.array(data['color']))
            # import pdb; pdb.set_trace()
            # states = get_state(color)
    
            # actions = get_action(color)

            object_poses = get_object_poses(data['info'])

            poses = get_pose(data['action'])

            # if transform:
            #     states = transform(states)
            #     actions = transform(actions)

            if self.cache:
                data.pop('color', None)
                data.pop('states', None)
                # data['states'] = states
                # data['action_img'] = actions
                data['object_poses'] = object_poses
                # self.data_list.append(data)
                self.data_list[idx] = data
        
                
        

        if self.seq_length is not None:
            idx_list = sample_idx(poses.shape[0], self.seq_length)
            # states = states[idx_list,...]
            # actions = actions[idx_list,...]
            poses = poses[idx_list,...]
            object_poses = object_poses[idx_list,...]
        # states = torch.zeros(poses.shape[0], 2,2)
        
        # output = {"state":states, "object_poses":object_poses, "pose": poses, "dpi": convert2dpi(states, object_poses, poses, self.cfg)}
        # output = {"state":states, "object_pose":object_poses, "pose": poses, "dpi": convert2dpi(object_poses, poses, self.cfg)}
        # output = {"state":states, "object_pose":object_poses, "pose": poses, 'idx':torch.tensor([idx])} #, "state":states, }
        output = {"object_pose":object_poses, "pose": poses, 'idx':torch.tensor([idx])}
        return output


def test_DynamicsDataset():
    from nfd.utils import utils
    cfg = utils.get_conf(name='dpi')
    training_set = DynamicsDataset(mode='val',index_max = None, name_filter=None, shuffle=False, seq_length = 20, data_dir='./data', cache=True, cfg = cfg)
    print(training_set[10]['pose'].shape)
    print(training_set[10]['dpi'][-1].shape)

    dataloader = DataLoader(training_set, batch_size=3, shuffle=True, num_workers=1, collate_fn=my_collate_all)

    for data_batch in dataloader:
        # print(data_batch['dpi'][0].shape)
        break
    
    # print(training_set[10].tolist())
    pass


if __name__ == '__main__':
    test_DynamicsDataset()