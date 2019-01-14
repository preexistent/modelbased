import numpy as np
from math import pow, atan2, sqrt, cos, sin, atan, asin
import dubins

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # physical angle velocity
        self.p_angle_vel = None
        #phisical rotation
        self.p_angle = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None
        # phisical rotation
        self.rot = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

    @property
    def momentum_mass(self):
        # return np.pi/4*self.size**4*self.initial_mass
        return 1.0

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # rotation range
        self.rot_range = 1.0
        # RVO
        self.RVO = {}
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        ##########################
        self._rel_heading = {}
        self.point_to_agent_heading = {}
        self._omega = {}
        self.VX = {}
        self.present_temp_h = None
        self.NR = 0.5
        self._distance = None
        self._least_distance = 10
        self.num = None
        self.den = None
        # update global agent information
        self.all_agents_pose_dict = {}


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # rotation dimensionality
        self.dim_rot = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1

        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        p_rot_force = [None]*len(self.entities)
        # apply agent physical controls
        p_force, p_rot_force = self.apply_action_force(p_force, p_rot_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force, p_rot_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # update global agent and obstacles information
    def update_agents_msg(self, agent):
        for cur_agent in self.agents:
            if agent.name != cur_agent.name:
                pose_updated = [cur_agent.state.p_pos[0], cur_agent.state.p_pos[1], atan2(cur_agent.state.p_vel[1],cur_agent.state.p_vel[0]),
                                np.sqrt(np.sum(np.asarray(cur_agent.state.p_vel) ** 2))]
                agent.all_agents_pose_dict.update({cur_agent.name: pose_updated})
        for obst in self.landmarks:
            pose_updated = [obst.state.p_pos[0], obst.state.p_pos[1], 0, 0]
            agent.all_agents_pose_dict.update({obst.name: pose_updated})

    def update_rvo(self, agent):
        self.update_agents_msg(agent)
        # define velocity magnitude
        v_mag = sqrt(pow((agent.state.p_vel[0]), 2) + pow((agent.state.p_vel[1]), 2))
        rr = 4
        r = 0.28
        #calc the relative velocity of the agent and choosen other agent
        agent._rel_heading = {}
        agent.point_to_agent_heading = {}
        agent._omega = {}
        agent.VX = {}
        angle = atan2(agent.state.p_vel[1], agent.state.p_vel[0])
        agent.present_temp_h = round(angle,rr)

        #neighbouring region is a circle of 3 units

        agent._least_distance = 10

        for i, cur_agent in enumerate(self.entities):
            if(cur_agent.name != agent.name) and "goal" not in cur_agent.name:
                #calc distance between agent and oher agent/obstacle
                name = cur_agent.name
                # print(cur_agent.name, agent.all_agents_pose_dict[name][0] - agent.state.p_pos[0],agent.all_agents_pose_dict[name][1] - agent.state.p_pos[1],
                #       round(sqrt(pow((agent.all_agents_pose_dict[name][0] - agent.state.p_pos[0]), 2) + pow(
                #           (agent.all_agents_pose_dict[name][1] - agent.state.p_pos[1]), 2)), rr)
                #       )
                agent._distance = round(sqrt(pow((agent.all_agents_pose_dict[name][0] - agent.state.p_pos[0]), 2) + pow((agent.all_agents_pose_dict[name][1] - agent.state.p_pos[1]), 2)),rr)

                #if it lies in the NR, consider it in calculating RVO
                if(agent._distance < agent.NR):
                    #calc the relative velocity of the agent and choosen other agent
                    agent._rel_v_x =  v_mag * cos(angle) - agent.all_agents_pose_dict[name][3] * cos(agent.all_agents_pose_dict[name][2])
                    agent._rel_v_y =  v_mag * sin(angle) - agent.all_agents_pose_dict[name][3] * sin(agent.all_agents_pose_dict[name][2])
                    agent._rel_heading[name] = round(atan2(agent._rel_v_y,agent._rel_v_x),rr)

                    # VO finder :: Should output a range of headings into an 2D array
                    agent.point_to_agent_heading[name] = round(atan2((agent.all_agents_pose_dict[name][1] - agent.state.p_pos[1]),(agent.all_agents_pose_dict[name][0] - agent.state.p_pos[0])),rr)
                    #can also use np.clip
                    try:
                        agent._omega[name] = round(asin(r/(agent._distance)),rr)
                        #print(cur_agent.name, agent._omega[name], agent._distance)
                    except ValueError:
                        agent._omega[name] = round(np.pi/2,rr)
                    # This is computationally easier

                    agent.VX[name] = (np.asarray([agent.point_to_agent_heading[name] - agent._omega[name],
                                              agent.point_to_agent_heading[name] + agent._omega[name]]))
                    #####find v_A by adding v_B to VX (both mag and dir)
                    # self.VO[i] = self.VX[i] + self.all_agents_pose_dict[i][2]
                    agent.num = v_mag * sin(agent.present_temp_h) - agent.all_agents_pose_dict[name][3] * sin(
                        agent.all_agents_pose_dict[name][2])
                    agent.den = v_mag * cos(agent.present_temp_h) - agent.all_agents_pose_dict[name][3] * cos(
                        agent.all_agents_pose_dict[name][2])


                    if 'obstacle' in name or agent.all_agents_pose_dict[name][3]< 0.02:#0.02
                        agent.RVO[name] = (agent.VX[name] + 0 * atan2(agent.num, agent.den)) / 1
                    else:
                        agent.RVO[name] = (agent.VX[name] + atan2(agent.num, agent.den)) / 2

                    # Uncomment the below line if you want the code to behave like VO
                    #agent.RVO[name] = agent.VX[name]

                    if (agent._distance < agent._least_distance):
                        agent._least_distance = agent._distance

        v_mag = (agent._least_distance/3)/agent.NR*1.5
        v_mag = min(v_mag, 0.8)

        return v_mag

    # Returns True when called if the agent is on collision course
    def collision(self, agent):
        angle = atan2(agent.state.p_vel[1], agent.state.p_vel[0])
        if(self.in_RVO(angle, agent) == True):
            #if True, return True. Else, False.
            return True
        return False

    def in_RVO(self,h , agent):
        #use sets for optimized code using "if h in self.RVO"
        #print(agent.RVO)
        for i in agent.RVO:
            if(agent.RVO[i][0] < h < agent.RVO[i][1]):
                return True
                break
        return False
        # Returns a new velocity that is outside RVO

    def choose_new_velocity_RVO(self, agent, goal_pose_x, goal_pose_y, agent_pose_x, agent_pose_y):
        rr = 2
        incr = 0.01
        desired_heading = atan2(goal_pose_y - agent_pose_y, goal_pose_x - agent_pose_x)
        _headings_array = np.round(np.arange(-np.pi, np.pi, incr), rr)

        # if not available, self.inside will return None.
        best_min = None

        # Find the nearest heading that is outside the VO
        temp_array_marginals = np.array([])

        for i in agent.RVO:
            temp_array_marginals = np.append(temp_array_marginals, agent.RVO[i])
            # self.temp_temp_temp = self.RVO[i][0]
        _h = np.round(temp_array_marginals, rr)

        # defining possible headings with a resolution of 0.01
        for i in range(len(_h)):
            if (i % 2 == 0):
                k = _h[i] + incr
                while (k < np.round(_h[i + 1], rr)):
                    # if(len(self._h) >1):
                    _headings_array = np.delete(_headings_array,
                                                     np.where(_headings_array == np.round(k, rr)))
                    k += incr
        # choosing heading nearest to goal heading
        # self._min_time_collision = self.time_to_collision(min(self.time_to_collision, key = self.time_to_collision.get))
        # self._min_time_collision = min(self.time_to_collision.items(), key=lambda x: x[1])
        idx = np.abs(_headings_array - desired_heading).argmin()
        # self.idx = (np.abs(self._headings_array - self.desired_heading) + 0.01/(self._min_time_collision+0.0001)).argmin()
        # choose whether left or right side is the nearest and then assign

        best_min = _headings_array[(idx - 1) % len(_headings_array)]
        """print("RVO is :")
        print(_headings_array)
        print("===")
        print("desired heading :")
        print(desired_heading)
        print("choosen direction is")
        print(best_min)
        print("===")
        """

        # rospy.sleep(1)
        #####then return a velocity that is average of current velocity and a velocity outside VO nearer to current heading
        return best_min
    # gather agent action forces
    def apply_action_force(self, p_force, p_rot_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
                p_rot_force[i] = agent.action.rot + noise

        return p_force, p_rot_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force, p_rot_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            entity.state.p_angle_vel = entity.state.p_angle_vel*(1-self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                entity.state.p_angle_vel += (p_rot_force[i] / entity.momentum_mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_angle += (entity.state.p_angle_vel * self.dt)
            if entity.state.p_angle > 0:
                entity.state.p_angle = entity.state.p_angle%(2*np.pi)
            else:
                entity.state.p_angle = -(-entity.state.p_angle%(2*np.pi)) + 2*np.pi

    """def integrate_state(self, p_force, p_rot_force):
        for i, entity in enumerate(self.entities):

            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            entity.state.p_angle_vel = entity.state.p_angle_vel * (1 - self.damping)
            if (p_force is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                entity.state.p_angle_vel += (p_rot_force[i] / entity.momentum_mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed

            q0 = (entity.state.p_pos[0], entity.state.p_pos[1], entity.state.p_angle)
            q1 = (entity.state.p_vel[0] * self.dt, entity.state.p_vel[1] * self.dt, entity.state.p_angle_vel * self.dt)
            r = entity.size
            s = 10

            path = dubins.shortest_path(q0, q1, r)
            configurations, _ = path.sample_many(s)
            #pdb.set_trace()

            for i in range(len(configurations)):
                entity.state.p_pos[0] += configurations[i][0]
                entity.state.p_pos[1] += configurations[i][1]
                entity.state.p_angle += configurations[i][2]
            if entity.state.p_angle > 0:
                entity.state.p_angle = entity.state.p_angle % (2 * np.pi)
            else:
                entity.state.p_angle = -(-entity.state.p_angle % (2 * np.pi)) + 2 * np.pi

            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_angle += (entity.state.p_angle_vel * self.dt)
            # if entity.state.p_angle > 0:
            #    entity.state.p_angle = entity.state.p_angle%(2*np.pi)
            # else:
            #    entity.state.p_angle = -(-entity.state.p_angle%(2*np.pi)) + 2*np.pi
        """
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]