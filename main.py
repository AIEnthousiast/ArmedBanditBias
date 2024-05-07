import numpy as np
import random 
import matplotlib.pyplot as plt

n_steps_per_run = 10000
n_runs = 500
e = 0.1
alpha = 0.1
k = 10


mean_rew_epsilon_av_sample = [0 for _ in range(n_steps_per_run)]

mean_rew_epsilon_const_alpha = [0 for _ in range(n_steps_per_run)]


for i in range(n_runs):
    print(f"run {i}")
    q_star = [1 for _ in range(k)]

    Q_epsilon_av_sample = [0 for _ in range(k)]
    choice_epsilon_av_sample = [0 for _ in range(k)]

    Q_epsilon_const_alpha = [0 for _ in range(k)]


    for j in range(n_steps_per_run):
        #greedy
        for l in range(k):
            q_star[l] += np.random.normal(loc=0,scale=0.01)
    
        if random.random() < e:
            a = random.randint(0,k-1)
            
        else:
            a = np.argmax(Q_epsilon_av_sample)

        r = np.random.normal(loc=q_star[a],scale=1)
        choice_epsilon_av_sample[a] +=1
        Q_epsilon_av_sample[a] = Q_epsilon_av_sample[a] + 1/(choice_epsilon_av_sample[a]) * (r - Q_epsilon_av_sample[a])
        mean_rew_epsilon_av_sample[j] = mean_rew_epsilon_av_sample[j] + 1/(i+1) * (r - mean_rew_epsilon_av_sample[j])

        if random.random() < e:
            a2 = random.randint(0,k-1)
        else:
            a2 = np.argmax(Q_epsilon_const_alpha)

        r = np.random.normal(loc=q_star[a2],scale=1)
        Q_epsilon_const_alpha[a2] = Q_epsilon_const_alpha[a2] + alpha * (r - Q_epsilon_const_alpha[a2])
        mean_rew_epsilon_const_alpha[j] = mean_rew_epsilon_const_alpha[j] + 1/(i+1) * (r - mean_rew_epsilon_const_alpha[j]) 

plt.plot(mean_rew_epsilon_const_alpha,label=f"constant alpha = {alpha}")
plt.plot(mean_rew_epsilon_av_sample,label=f"average sample")
plt.legend()
plt.show()