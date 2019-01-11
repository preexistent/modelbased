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
        world.num_agents = 4
        world.num_goals = 4
        # world.num_obstacles = 2
        world.num_obstacles = 2
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
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                landmark.name = 'goal %d' %i
                landmark.collide = False
                landmark.movable = False
                landmark.state.p_vel = np.random.normal(size=2)
            else:
                landmark.name = 'obstacle %d' %(i-world.num_goals)
                landmark.collide = True
                landmark.movable = False
        self.reset_world(world)
        return world
    
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.random.normal(size=2)
            agent.state.c = np.zeros(world.dim_c)
            agent.color = np.array([0.35, 0., 0.])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            if i <world.num_goals:
                landmark.color = np.array([0., .35, 0.])
                landmark.state.p_vel = np.random.normal(size=2)
                '''
                Initialize velocity vectors for different landmarks
                landmark.p_vel = np.array([1,1])
                '''
            else:
                landmark.color = np.array([0., 0., .35])
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
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        coef_collision = 1.1
        coef_dist = 1.0
        coef_cosdist = 0.
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                rew -= min(dists) * coef_dist
            else:
                if agent.collide:
                    if self.is_collision(agent, landmark):
                        rew -= 1 * coef_collision
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1 * coef_collision
        '''
            We can modify this part to include velocity direction into the model.
            Compute cosine distance between agent velocity and landmark velocity.
            distance = spatial.distance.cosine(agent.p_vel, landmark.p_vel);
            As the distance increase, the reward should decrease, so we should 
            rew -=  const*distance.
        '''
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                # cos_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                cos_dist = [spatial.distance.cosine(a.state.p_vel, landmark.state.p_vel) for a in world.agents]
                rew -= min(cos_dist) * coef_cosdist
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)