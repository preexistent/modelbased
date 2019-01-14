import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from scipy import spatial
import math

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = 3
        world.num_goals = 3
        # world.num_obstacles = 2
        world.num_obstacles = 3
        world.collaborative = True
        # self.landmarkspeed = np.random.normal(size=2)
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add goals
        world.landmarks = [Landmark() for i in range(world.num_goals+world.num_obstacles)]
        p_vel = np.zeros(2)
        p_angle = np.random.uniform(size=1)
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                landmark.name = 'goal %d' %i
                landmark.collide = False
                landmark.movable = False
                landmark.state.p_vel = p_vel
                landmark.state.p_angle = p_angle
            else:
                landmark.size = 0.12
                landmark.name = 'obstacle %d' %(i-world.num_goals)
                landmark.collide = True
                landmark.movable = False
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        pos_array = assign_pos( world.num_agents+world.num_goals+world.num_obstacles)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = pos_array[i]
            agent.state.p_vel = np.zeros(2)
            # for rotation
            agent.state.p_angle = np.random.uniform(0, 2*math.pi,1)
            agent.state.p_angle_vel =0
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, i/10, 0.])
            agent.max_speed = 0.3

        # random properties for landmarks
        landmark_angle = np.random.uniform(0, 2*math.pi, 1)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = pos_array[world.num_agents+i]
            if i <world.num_goals:
                landmark.color = np.array([0., .35, 0.])
                landmark.color_ind = np.array([0,1])
                landmark.state.p_angle = landmark_angle
                '''
                Initialize velocity vectors for different landmarks
                landmark.p_vel = np.array([1,1])
                '''
            else:
                landmark.color = np.array([0., 0., .35])
                landmark.color_ind = np.array([1,0])
                landmark.size = 0.12

        # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)
    
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.1:
                    occupied_landmarks += 1
            else:
                if agent.collide:
                    if self.is_collision(a, landmark):
                        rew -= 1
                        collisions += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)
    
    def is_collision(self, agent1, agent2):
        # delta_pos = agent1.state.p_pos - agent2.state.p_pos
        # dist = np.sqrt(np.sum(np.square(delta_pos)))
        # dist_min = agent1.size + agent2.size
        # return True if dist < dist_min else False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        if dist <= dist_min:
            return 1
        else:
            return 0
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        rew_dist = 0
        rew_collision = 0
        rew_cos_dist = 0
        coef_collision = 1
        coef_dist = 1
        coef_cosdist = 1
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                rew_dist -= min(dists) * coef_dist
            else:
                if agent.collide:
                    # if self.is_collision(agent, landmark):
                    rew_collision -= self.is_collision(agent, landmark) * coef_collision
        if agent.collide:
            for a in world.agents:
                # if self.is_collision(a, agent):
                rew_collision-= self.is_collision(a, agent) * coef_collision
        '''
            We can modify this part to include velocity direction into the model.
            Compute cosine distance between agent velocity and landmark velocity.
            distance = spatial.distance.cosine(agent.p_vel, landmark.p_vel);
            As the distance increase, the reward should decrease, so we should 
            rew -=  const*distance.
        '''
        # for i, landmark in enumerate(world.landmarks):
        #     if i <world.num_goals:
        #         # cos_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
        #         cos_dist = [spatial.distance.cosine(a.state.p_vel, landmark.state.p_vel) for a in world.agents]
        #         rew -= min(cos_dist) * coef_cosdist
        
        cos_dist = []
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                # cos_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                diff_angle = abs(agent.state.p_angle-landmark.state.p_angle)
                cos_dist.append(np.pi - abs(diff_angle-np.pi))
        rew_cos_dist -= min(cos_dist) * coef_cosdist

        rew = rew_collision + rew_cos_dist + rew_dist

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            # if i <world.num_goals:
            #     entity_vel.append(entity.state.p_vel-agent.state.p_vel)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color_ind)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # for rotation
        entity_angle=[]
        for i, entity in enumerate(world.landmarks):
            if i < world.num_goals:
                #print("entity angle : {}, agent angle : {}".format(entity.state.p_angle, agent.state.p_angle))
                temp_angle = np.append(entity.state.p_angle-agent.state.p_angle, 0)
                entity_angle.append(temp_angle)

        return np.concatenate([agent.state.p_vel] + entity_angle +entity_color+[agent.state.p_pos] + [agent.state.p_angle] + entity_pos +  other_pos )
# def assign_pos(number):
#     pos = np.zeros((number,2))
#     pos_y_choice = np.linspace(-1,1,num=number)
#     pos[:,1] = np.random.choice(pos_y_choice, number,replace=False)
#     for i in range(1,number):
#         dist = np.zeros(i)
#         while(np.any(dist<=0.35**2)):
#             pos[i,0]= np.random.uniform(-1,1,size=1)
#             for j in range(0,i):
#                 dist[j]=(np.sum(np.square(pos[i]-pos[j])))
#     return pos
#
def assign_pos(number):
    pos = np.zeros((number, 2))
    pos_x_choice = np.linspace(-1, 1, num=2 / 0.25)
    pos_y_choice = np.linspace(-1, 1, num=2 / 0.25)

    x, y = np.meshgrid(pos_x_choice, pos_y_choice)
    choice = np.arange(x.size)
    target = np.random.choice(choice, number, replace=False)
    for i in range(number):
        pos[i] = np.asarray((x.flatten()[target[i]], y.flatten()[target[i]]))
    return pos