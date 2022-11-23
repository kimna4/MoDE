import os
import scipy
import scipy.misc

import torch
import numpy as np
import time

# from carla.agent import Agent
# from carla.carla_server_pb2 import Control

# from agents.representation.modules.ACTION_MINE.mine_net_posi50 import CarlaDisentangled
from agents.representation.modules.ACTION_MINE.mine_net_posi50_v2_gating import CarlaDisentangled
from agents.representation.modules.ACTION_MINE.VAE_net_lusr import CarlaDisentangledVAE

from version084.benchmark_tools.agent import Agent
from version084.carla.carla_server_pb2 import Control
import torch.nn.functional as F

from utils.representation_posi50_gating_adj import action_adjusting_carla100

class ImitationLearning(Agent):

    def __init__(self, city_name,
                 avoid_stopping=True, model_path="model/policy.pth",
                 visualize=False, log_name="test_log",
                 class_latent_size=64, content_latent_size=128, vae_model_path="model.pth",
                 image_cut=[125, 510]
                 ):

        super(ImitationLearning, self).__init__()
        # Agent.__init__(self)

        self._image_size = (256, 256, 3)
        self._avoid_stopping = avoid_stopping
        self.vae_model_path = vae_model_path

        dir_path = os.path.dirname(__file__)
        self._models_path = os.path.join(dir_path, model_path)
        self.model = CarlaDisentangled(encoder_name='Basic', class_latent_size=class_latent_size,
                              content_latent_size=content_latent_size)
        self.model_VAE = CarlaDisentangledVAE(class_latent_size=class_latent_size,
                                         content_latent_size=content_latent_size)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model_VAE = torch.nn.DataParallel(self.model_VAE).cuda()

        self.load_model()
        self.model.eval()
        self.load_model_vae()
        self.model_VAE.eval()

        self._image_cut = image_cut

        # by kimna
        self.before_episode_name = ''
        self.before_img = []
        self.stopping_cnt = 0
        self.running_cnt = 0
        self.before_direction = 0
        self.before_steer = 0

    def load_model(self):
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path: %s'
                               % self._models_path)
        # checkpoint = torch.load(self._models_path, map_location='cuda:0')
        checkpoint = torch.load(self._models_path)
        now_state_dict = self.model.state_dict()
        pretrained_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in now_state_dict}
        now_state_dict.update(pretrained_state_dict)
        self.model.load_state_dict(now_state_dict)

    def load_model_vae(self):
        vae_model_path = self.vae_model_path
        # pretrained_state_dict = torch.load(vae_model_path, map_location='cuda:0')
        pretrained_state_dict = torch.load(vae_model_path)
        now_state_dict = self.model_VAE.state_dict()
        # 1. filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict['state_dict'].items() if k in now_state_dict}
        # 2. overwrite entries in the existing state dict
        now_state_dict.update(pretrained_state_dict)
        # 3. load the new state dict
        self.model_VAE.load_state_dict(now_state_dict)

    def run_step(self, measurements, sensor_data, directions, target, episode_name):

        # by kimna
        # When Episode Init
        if self.before_episode_name == '':
            self.before_episode_name = episode_name
            self.stopping_cnt = 0
            self.running_cnt = 0
            self.before_steer = 0
        elif not self.before_episode_name == episode_name:
            self.before_episode_name = episode_name
            self.stopping_cnt = 0
            self.running_cnt = 0
            self.before_steer = 0

        start = time.time()
        control = self._compute_action(
            sensor_data['CameraRGB'].data,
            measurements, episode_name,
            directions)
        # print('time: ', time.time() - start)
        return control

    def _compute_action(self, rgb_image, measurements, episode_name, direction=None):

        speed = measurements.player_measurements.forward_speed
        loc_x = measurements.player_measurements.transform.location.x
        loc_y = measurements.player_measurements.transform.location.y
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.expand_dims(np.transpose(image_input, (2, 0, 1)), axis=0)

        image_input = np.multiply(image_input, 1.0 / 255.0)
        speed = np.array([[speed]]).astype(np.float32) * 3.6

        # speed = np.array([[speed]]).astype(np.float32) / 10 # by kimna
        direction = int(direction-2)
        if direction == -2:
            direction = 0

        if self.before_direction == 1 and direction == 2:
            direction = 0
        elif self.before_direction == 2 and direction == 1:
            direction = 0
        elif self.before_direction == 1 and direction == 3:
            direction = 0
        elif self.before_direction == 2 and direction == 3:
            direction = 0
        else:
            self.before_direction = direction


        # ImitationLearning.beforeImg = image_input

        steer, acc, brake, pred_speed = self._control_function(image_input, speed / 40, direction)

        self.running_cnt += 1

        if speed < 1:
            self.stopping_cnt += 1
        else:
            self.stopping_cnt = 0

        ori_str = steer
        ori_brake = brake
        ori_acc = acc
        steer, acc, brake, self.stopping_cnt = action_adjusting_carla100(direction, steer, acc, brake, speed,
                                                                         pred_speed, self.running_cnt, self.stopping_cnt)

        print(
            '**[%d, %d] direc: %d, steer: %.3f (%.3f), break: %.5f (%.5f), acc: %.5f (%.5f), real speed: %.3f, pred_speed: %.3f, stopping_cnt: %d '
            % (loc_x, loc_y, direction, steer, ori_str, brake, ori_brake, acc, ori_acc, speed, pred_speed,
               self.stopping_cnt))

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0
        self.before_steer = steer

        return control

    def _control_function(self, image_input, speed, control_input):

        img_ts = torch.from_numpy(image_input).cuda()
        speed_ts = torch.from_numpy(speed).cuda()

        with torch.no_grad():
            mu, logsigma, classcode = self.model_VAE(img_ts)
            # branches, pred_speed = self.model(img_ts, speed_ts, mu, classcode, 0)
            weat_out, pred_speed, act_output1, act_output2, act_output3, act_output4\
                = self.model(img_ts, speed_ts, mu, logsigma, classcode)


        pred_weat = F.softmax(weat_out, dim=1).cpu().numpy()
        pred_result1 = act_output1[0][3 * control_input:3 * (control_input + 1)].cpu().numpy() * pred_weat[0][0]
        pred_result2 = act_output2[0][3 * control_input:3 * (control_input + 1)].cpu().numpy() * pred_weat[0][1]
        pred_result3 = act_output3[0][3 * control_input:3 * (control_input + 1)].cpu().numpy() * pred_weat[0][2]
        pred_result4 = act_output4[0][3 * control_input:3 * (control_input + 1)].cpu().numpy() * pred_weat[0][3]

        pred_result = pred_result1 + pred_result2 + pred_result3 + pred_result4


        predicted_steers = (pred_result[0])
        predicted_acc = (pred_result[1])
        predicted_brake = (pred_result[2])

        predicted_speed = pred_speed.squeeze().item()

        return predicted_steers, predicted_acc, predicted_brake, predicted_speed * 40.0
