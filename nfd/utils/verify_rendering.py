from nfd.agents.BaseAgent import BaseAgent, get_action
from nfd.agents.LineAgent import LineShootAgent, LineOptAgent, gen_pose_list_batch

from nfd.eval.Evaluator import Evaluator

from nfd.utils.utils import get_conf
import torch
import matplotlib.pyplot as plt


class Verify(Evaluator):
    def eval(self):
        
        env = self.env
        env.set_task(self.task)
        # task = self.task
        # env.step(target)
        obs = env.reset()
        img = self.get_image(obs)
        info = env.info
        
        while True:
            endpoints = self.agent_eval.sampler(1)
            traj = gen_pose_list_batch(torch.tensor(endpoints), horizon=1)[0,...]

            target = self.agent_eval.post_process(traj)

            obs, reward, done, info = env.step(target[1])
            state, color, goal_zone = self.obs_process(obs, info)

            plt.subplot(1,2,1)
            self.visualize_field(traj[0,...], self.pusher_field)
            plt.subplot(1,2,2)
            self.visualize_obs(get_action(color))
            plt.pause(0.001)
            input()


def main():
    cfg = get_conf()
    vv = Verify(cfg)
    vv.set_agent(LineShootAgent(cfg))
    vv.eval()

if __name__ == '__main__':
    main()