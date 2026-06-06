import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import copy

# ==========================================
# THIẾT LẬP THIẾT BỊ (GPU/CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Đang chạy trên thiết bị: {device}")

# ==========================================
# 1. KIẾN TRÚC MẠNG CHUNG
# ==========================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))

# ==========================================
# 2. REPLAY BUFFER
# ==========================================
class ReplayBuffer(object):
    def __init__(self, max_size=200000):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in ind:
            s, a, r, s_, d = self.storage[i]
            states.append(s); actions.append(a); rewards.append(r)
            next_states.append(s_); dones.append(d)
        return (
            np.array(states), np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states), np.array(dones).reshape(-1, 1)
        )

# ==========================================
# 3. THUẬT TOÁN DDPG & TD3
# ==========================================
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_q_value(self, state, action):
        """Trả về Q-value ước lượng cho cặp (state, action)."""
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_t = torch.FloatTensor(action.reshape(1, -1)).to(device)
        with torch.no_grad():
            q = self.critic(state_t, action_t)
        return q.item()

    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        not_done = torch.FloatTensor(1.0 - done).to(device)

        # DDPG: dùng 1 critic duy nhất → dễ overestimate
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * discount * target_Q).detach()

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Cập nhật Actor mỗi step (không delay)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return current_Q.mean().item()

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def get_q_value(self, state, action):
        """Trả về Q1-value ước lượng cho cặp (state, action)."""
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_t = torch.FloatTensor(action.reshape(1, -1)).to(device)
        with torch.no_grad():
            q = self.critic.Q1(state_t, action_t)
        return q.item()

    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        not_done = torch.FloatTensor(1.0 - done).to(device)

        with torch.no_grad():
            # TD3: Target Policy Smoothing
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            # TD3: Clipped Double-Q → lấy min(Q1, Q2) để giảm overestimation
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (not_done * discount * torch.min(target_Q1, target_Q2))

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # TD3: Delayed Policy Updates
        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return current_Q1.mean().item()


# ==========================================
# 4. TÍNH TRUE Q-VALUE BẰNG MONTE CARLO ROLLOUT
# ==========================================
def compute_true_q_value(env_name, agent, num_episodes=10, discount=0.99):
    """
    Chạy agent trong env, ghi nhận (s0, a0) ở bước đầu và 
    tính True Q = sum(gamma^t * r_t) cho toàn bộ episode.
    Trả về: trung bình estimated_Q, trung bình true_Q.
    """
    env = gym.make(env_name)
    estimated_q_list = []
    true_q_list = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        action = agent.select_action(np.array(state))

        # Estimated Q tại (s0, a0)
        est_q = agent.get_q_value(np.array(state), np.array(action))
        estimated_q_list.append(est_q)

        # Rollout để tính True Q (Monte Carlo return)
        rewards = []
        done = False
        s = state
        a = action
        first_step = True
        while not done:
            if first_step:
                next_state, reward, terminated, truncated, _ = env.step(a)
                first_step = False
            else:
                a = agent.select_action(np.array(s))
                next_state, reward, terminated, truncated, _ = env.step(a)
            rewards.append(reward)
            done = terminated or truncated
            s = next_state

        # Tính discounted return
        true_q = 0.0
        for r in reversed(rewards):
            true_q = r + discount * true_q
        true_q_list.append(true_q)

    env.close()
    return np.mean(estimated_q_list), np.mean(true_q_list)


# ==========================================
# 5. HÀM HUẤN LUYỆN VÀ THU THẬP OVERESTIMATION BIAS
# ==========================================
def train_agent(agent_name, AgentClass, env_name="Pendulum-v1",
                max_timesteps=100000, start_steps=1000,
                eval_freq=2000, batch_size=256, discount=0.99):
    """
    Huấn luyện agent và định kỳ đo Overestimation Bias.
    Trả về: eval_steps, biases, estimated_qs, true_qs, reward_history
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = AgentClass(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    expl_noise = 0.1

    # Logs
    eval_steps = []        # Timestep tại mỗi lần đánh giá
    biases = []            # Overestimation Bias = Estimated Q - True Q
    estimated_qs = []      # Estimated Q trung bình
    true_qs = []           # True Q trung bình
    reward_history = []    # Reward mỗi episode
    episode_indices = []   # Timestep kết thúc mỗi episode

    state, _ = env.reset()
    episode_reward = 0
    episode_num = 0
    total_timesteps = 0

    print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN: {agent_name} trên {env_name}...")
    print(f"   Tổng steps: {max_timesteps}, Eval mỗi: {eval_freq} steps\n")

    while total_timesteps < max_timesteps:
        total_timesteps += 1

        # Chọn action
        if total_timesteps < start_steps:
            action = env.action_space.sample()
        else:
            action = (
                agent.select_action(np.array(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, float(terminated))
        state = next_state
        episode_reward += reward

        # Train
        if total_timesteps >= start_steps:
            agent.train(replay_buffer, batch_size=batch_size, discount=discount)

        # Kết thúc episode
        if done:
            episode_num += 1
            reward_history.append(episode_reward)
            episode_indices.append(total_timesteps)

            if episode_num % 20 == 0:
                print(f"  [{agent_name}] Episode: {episode_num} | "
                      f"Steps: {total_timesteps} | "
                      f"Reward: {episode_reward:.1f}")

            state, _ = env.reset()
            episode_reward = 0

        # Định kỳ đánh giá Overestimation Bias
        if total_timesteps % eval_freq == 0 and total_timesteps >= start_steps:
            est_q, true_q = compute_true_q_value(
                env_name, agent, num_episodes=10, discount=discount
            )
            bias = est_q - true_q

            eval_steps.append(total_timesteps)
            biases.append(bias)
            estimated_qs.append(est_q)
            true_qs.append(true_q)

            print(f"  📊 [{agent_name}] Step {total_timesteps}: "
                  f"Est_Q={est_q:.2f} | True_Q={true_q:.2f} | "
                  f"Bias={bias:.2f}")

    env.close()
    print(f"\n✅ {agent_name} hoàn thành! Tổng episodes: {episode_num}")
    return eval_steps, biases, estimated_qs, true_qs, reward_history


# ==========================================
# 6. HÀM LÀM MỊN (EXPONENTIAL MOVING AVERAGE)
# ==========================================
def smooth_curve(scalars, weight=0.9):
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ==========================================
# 7. CHẠY VÀ VẼ BIỂU ĐỒ SO SÁNH
# ==========================================
if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    MAX_TIMESTEPS = 100000
    EVAL_FREQ = 2000
    DISCOUNT = 0.99

    # 1. Chạy DDPG
    ddpg_steps, ddpg_bias, ddpg_est_q, ddpg_true_q, ddpg_reward = train_agent(
        "DDPG", DDPG, env_name=ENV_NAME,
        max_timesteps=MAX_TIMESTEPS, eval_freq=EVAL_FREQ, discount=DISCOUNT
    )

    # 2. Chạy TD3
    td3_steps, td3_bias, td3_est_q, td3_true_q, td3_reward = train_agent(
        "TD3", TD3, env_name=ENV_NAME,
        max_timesteps=MAX_TIMESTEPS, eval_freq=EVAL_FREQ, discount=DISCOUNT
    )

    # ========== VẼ BIỂU ĐỒ ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Biểu đồ 1: Overestimation Bias (Estimated Q - True Q) ----
    ax1 = axes[0, 0]
    ax1.plot(ddpg_steps, ddpg_bias, label='DDPG Bias', color='red',
             linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax1.plot(td3_steps, td3_bias, label='TD3 Bias', color='blue',
             linewidth=2, marker='s', markersize=4, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Không Bias (y=0)')
    ax1.set_title('Overestimation Bias = Estimated Q − True Q', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Bias (Estimated Q − True Q)')
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.fill_between(ddpg_steps, 0, ddpg_bias, alpha=0.15, color='red')
    ax1.fill_between(td3_steps, 0, td3_bias, alpha=0.15, color='blue')

    # ---- Biểu đồ 2: Estimated Q vs True Q cho DDPG ----
    ax2 = axes[0, 1]
    ax2.plot(ddpg_steps, ddpg_est_q, label='DDPG Estimated Q', color='red',
             linewidth=2, linestyle='-')
    ax2.plot(ddpg_steps, ddpg_true_q, label='DDPG True Q (MC)', color='darkred',
             linewidth=2, linestyle='--')
    ax2.plot(td3_steps, td3_est_q, label='TD3 Estimated Q', color='blue',
             linewidth=2, linestyle='-')
    ax2.plot(td3_steps, td3_true_q, label='TD3 True Q (MC)', color='darkblue',
             linewidth=2, linestyle='--')
    ax2.set_title('Estimated Q vs True Q (Monte Carlo)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Q-Value')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ---- Biểu đồ 3: Episode Reward ----
    ax3 = axes[1, 0]
    ddpg_reward_smooth = smooth_curve(ddpg_reward, weight=0.9)
    td3_reward_smooth = smooth_curve(td3_reward, weight=0.9)
    eps_ddpg = range(1, len(ddpg_reward) + 1)
    eps_td3 = range(1, len(td3_reward) + 1)

    ax3.plot(eps_ddpg, ddpg_reward, color='red', alpha=0.15)
    ax3.plot(eps_td3, td3_reward, color='blue', alpha=0.15)
    ax3.plot(eps_ddpg, ddpg_reward_smooth, label='DDPG (Smoothed)', color='red',
             linewidth=2.5, linestyle='--')
    ax3.plot(eps_td3, td3_reward_smooth, label='TD3 (Smoothed)', color='blue',
             linewidth=2.5)
    ax3.set_title('Hiệu suất học tập (Episode Reward)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Reward')
    ax3.legend(fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ---- Biểu đồ 4: Tổng hợp Bias trung bình ----
    ax4 = axes[1, 1]
    avg_ddpg_bias = np.mean(ddpg_bias) if ddpg_bias else 0
    avg_td3_bias = np.mean(td3_bias) if td3_bias else 0
    bars = ax4.bar(['DDPG', 'TD3'], [avg_ddpg_bias, avg_td3_bias],
                   color=['red', 'blue'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Trung bình Overestimation Bias', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Average Bias')

    # Ghi số lên cột
    for bar, val in zip(bars, [avg_ddpg_bias, avg_td3_bias]):
        ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    plt.suptitle(f'So sánh Overestimation: DDPG vs TD3 trên {ENV_NAME}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('td3_vs_ddpg_overestimation.png', dpi=150, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: td3_vs_ddpg_overestimation.png")
    plt.show()