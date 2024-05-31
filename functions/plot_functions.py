import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

def plot_input_LLM_and_SF(results, t_start, t_end):
    time = np.arange(len(results[f'agent {str(len(results)-1)}']['acc pred SF']))
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(time[t_start:t_end], results[f'agent {str(len(results)-1)}']['acc pred SF'][t_start:t_end])
    axes[0].plot(time[t_start:t_end], results[f'agent {str(len(results) - 1)}']['acc pred LLM'][t_start:t_end])
    axes[0].set_title('Acceleration')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('[m/^2]')
    axes[0].legend(['SF', 'LLM'])

    SF = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred SF']]
    LLM = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred LLM']]
    axes[1].plot(time[t_start:t_end], SF[t_start:t_end])
    axes[1].plot(time[t_start:t_end], LLM[t_start:t_end])
    axes[1].set_title('Steering angle')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('[deg]')
    axes[1].legend(['SF', 'LLM'])

    plt.show()


def input_animation(results, t_start, t_end):
    fig, ax1= plt.subplots()

    time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    acc_SF = results[f'agent {str(len(results)-1)}']['acc pred SF']
    acc_LLM = results[f'agent {str(len(results) - 1)}']['acc pred LLM']

    line1, = ax1.plot(time[t_start], acc_SF[t_start], color='orange')
    line2, = ax1.plot(time[t_start], acc_LLM[t_start], color='red')

    ax1.set_xlim(min(time[t_start:t_end]) - 0.5, max(time[t_start:t_end]) + 0.5)
    ax1.set_ylim(min(min(acc_SF), min(acc_LLM)) - 0.5, max(max(acc_SF), max(acc_LLM)) + 0.5)

    ax1.legend(['SF', 'LLM'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('[m/s^2]')
    ax1.set_title('Acceleration')

    def update1(frame):
        line1.set_xdata(time[t_start:t_start+frame])
        line1.set_ydata(acc_SF[t_start:t_start+frame])
        line2.set_xdata(time[t_start:t_start+frame])
        line2.set_ydata(acc_LLM[t_start:t_start+frame])
        return line1, line2

    # Create animations
    ani1 = FuncAnimation(fig=fig, func=update1, frames=t_end-t_start, interval=100, blit=True, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani1.save(path + '/animation/acc_input.gif', writer='pillow')
    plt.show()

    fig, ax2 = plt.subplots()

    time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    steering_SF = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred SF']]
    steering_LLM = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred LLM']]

    line3, = ax2.plot(time[t_start], steering_SF[t_start], color='orange')
    line4, = ax2.plot(time[t_start], steering_LLM[t_start], color='red')

    ax2.set_xlim(min(time[t_start:t_end])  - 0.5, max(time[t_start:t_end]) + 0.5)
    ax2.set_ylim(min(min(steering_SF), min(steering_LLM)) - 0.5, max(max(steering_SF), max(steering_LLM)) + 0.5)

    ax2.legend(['SF', 'LLM'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('[deg]')
    ax2.set_title('Steering angle')

    def update2(frame):
        line3.set_xdata(time[t_start:t_start+frame])
        line3.set_ydata(steering_SF[t_start:t_start+frame])
        line4.set_xdata(time[t_start:t_start+frame])
        line4.set_ydata(steering_LLM[t_start:t_start+frame])

        return line3, line4

    ani2 = FuncAnimation(fig=fig, func=update2, frames=t_end-t_start, interval=100, blit=True, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani2.save(path + '/animation/steering_input.gif', writer='pillow')

    plt.show()

    return ani1, ani2

def prep_plot_vehicles(results, env, t_start, ax):

    vehicles = {}
    labels = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][t_start] * 180 / np.pi
            if results[f'agent {id_agent}']['type'] == 'standard car':
                color = 'green'
            elif results[f'agent {id_agent}']['type'] == 'emergency car':
                color = 'red'
            if id_agent != len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                     results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor=color, label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                                ha='center', va='center', color='white')
            else:
                if env['With LLM car']:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                         results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='blue', label='LLM car')
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                    results[f'agent {id_agent}']['y coord'][t_start], 'LLM',
                                                    ha='center', va='center', color='black')
                else:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                         results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor=color, label=str(id_agent))
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                    results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                                    ha='center', va='center', color='white')
            ax.add_patch(vehicles[f'{id_agent}'])

    lines = {}
    # scats = {}
    for k in range(len(results)):
        if env['With LLM car'] and k == len(results) - 1:
            lines[f'line{k} SF'] = \
                ax.plot(results[f'agent {k}']['x coord pred SF'][t_start],
                        results[f'agent {k}']['y coord pred SF'][t_start],
                        c="orange", linestyle='--')[0]  # label=f'v0 = {v02} m/s'
        else:
            lines[f'line{k} traj estimation'] = \
                ax.plot(results[f'agent {k}']['trajectory estimation x'][t_start],
                        results[f'agent {k}']['trajectory estimation y'][t_start], c="green",
                        linestyle='--')[0]  # label=f'v0 = {v02} m/s'
        lines[f'line{k}'] = \
            ax.plot(results[f'agent {k}']['x coord pred'][t_start], results[f'agent {k}']['y coord pred'][t_start],
                    c="red",
                    linestyle='--')[0]  # label=f'v0 = {v02} m/s'

    return vehicles, labels, lines, ax

def plot_vehicles(results, fig, ax, env, t_start, t_end):
    """vehicles = {}
    labels = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][t_start] * 180 / np.pi
            if results[f'agent {id_agent}']['type'] == 'standard car':
                color = 'green'
            elif results[f'agent {id_agent}']['type'] == 'emergency car':
                color = 'red'
            if id_agent != len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][t_start] - L / 2, results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor=color, label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                                ha='center', va='center', color='white')
            else:
                if env['With LLM car']:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][t_start] - L / 2, results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='blue', label='LLM car')
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                    results[f'agent {id_agent}']['y coord'][t_start], 'LLM',
                                                    ha='center', va='center', color='black')
                else:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][t_start] - L / 2, results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor=color, label=str(id_agent))
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                    results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                                    ha='center', va='center', color='white')
            ax.add_patch(vehicles[f'{id_agent}'])

    lines = {}
    # scats = {}
    for k in range(len(results)):
        if env['With LLM car'] and k == len(results) - 1:
            lines[f'line{k} SF'] = \
            ax.plot(results[f'agent {k}']['x coord pred SF'][t_start], results[f'agent {k}']['y coord pred SF'][t_start],
                    c="orange", linestyle='--')[0]  # label=f'v0 = {v02} m/s'
        else:
            lines[f'line{k} traj estimation'] = \
                ax.plot(results[f'agent {k}']['trajectory estimation x'][t_start],
                        results[f'agent {k}']['trajectory estimation y'][t_start], c="green",
                        linestyle='--')[0]  # label=f'v0 = {v02} m/s'
        lines[f'line{k}'] = \
            ax.plot(results[f'agent {k}']['x coord pred'][t_start], results[f'agent {k}']['y coord pred'][t_start], c="red",
                    linestyle='--')[0]  # label=f'v0 = {v02} m/s'

        # scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    vehicles, labels, lines, ax = prep_plot_vehicles(results, env, t_start, ax)
    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][t_start+frame] - \
                env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length'] / 2
            y = results[f'agent {id_agent}']['y coord'][t_start+frame] - \
                env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width'] / 2
            angle = results[f'agent {id_agent}']['theta'][t_start+frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)
            data = np.stack(
                [results[f'agent {id_agent}']['x coord'][t_start+frame], results[f'agent {id_agent}']['y coord'][t_start+frame]]).T
            labels[f'{id_agent}'].set_position(data)
            if env['With LLM car'] and id_agent == str(len(results) - 1):
                lines[f'line{id_agent} SF'].set_xdata(results[f'agent {id_agent}']['x coord pred SF'][t_start+frame])
                lines[f'line{id_agent} SF'].set_ydata(results[f'agent {id_agent}']['y coord pred SF'][t_start+frame])
            else:
                lines[f'line{id_agent} traj estimation'].set_xdata(
                    results[f'agent {id_agent}']['trajectory estimation x'][t_start+frame])
                lines[f'line{id_agent} traj estimation'].set_ydata(
                    results[f'agent {id_agent}']['trajectory estimation y'][t_start+frame])
            lines[f'line{id_agent}'].set_xdata(results[f'agent {id_agent}']['x coord pred'][t_start+frame])
            lines[f'line{id_agent}'].set_ydata(results[f'agent {id_agent}']['y coord pred'][t_start+frame])

        return (vehicles, labels, lines)

    ani = FuncAnimation(fig=fig, func=update, frames=t_end-t_start, interval=100, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani.save(path + '/animation/animation.gif', writer='pillow')

    return ani

def plot_simulation(env_type, env, results, t_start, t_end):

    fig, ax = plt.subplots()

    if env_type == 0:
        ani, ax, fig = plot_simulation_env_0(env, results, t_start, t_end, fig, ax)
    elif env_type == 1:
        ani, ax, fig = plot_simulation_env_1(env, results, t_start, t_end, fig, ax)
    elif env_type == 2:
        ani, ax, fig = plot_simulation_env_2(env, results, t_start, t_end, fig, ax)
    elif env_type == 3:
        ani, ax, fig = plot_simulation_env_3(env, results, t_start, t_end, fig, ax)
    elif env_type == 4:
        ani, ax, fig = plot_simulation_env_4(env, results, t_start, t_end, fig, ax)
    elif env_type == 5:
        ani, ax, fig = plot_simulation_env_5(env, results, t_start, t_end, fig, ax)
    else:
        print('Not ready')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani

def plot_simulation_env_0(env, results, t_start, t_end, fig, ax):

    obstacles = env['Road Limits']
    for id in obstacles:
        """if obstacles[id]['radius'][0] != obstacles[id]['radius'][1]:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif obstacles[id]['radius'][0] == obstacles[id]['radius'][1]:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)"""
        ax.set_aspect('equal')
        ax.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    ax.plot([-25, -6], [0, 0], color='black', linestyle='--')
    ax.plot([25, 6], [0, 0], color='black', linestyle='--')
    ax.plot([0, 0], [-25, -6], color='black', linestyle='--')
    ax.plot([0, 0], [25, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        ax.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        ax.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    if t_start != None:
        ani = plot_vehicles(results, fig, ax, env, t_start, t_end)
    else:
        ani = None

    #ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    #ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig


def plot_simulation_env_1(env, results, t_start, t_end, fig, ax):

    obstacles = env['Road Limits']
    for id in obstacles:
        """if obstacles[id]['radius'][0] != obstacles[id]['radius'][1]:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif obstacles[id]['radius'][0] == obstacles[id]['radius'][1]:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)"""
        ax.set_aspect('equal')
        plt.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    patch =  patches.Rectangle((0, -7), 6, 1, angle=0 , rotation_point='center', facecolor='black')
    ax.add_patch(patch)
    patch = patches.Rectangle((-6, 6), 6, 1, angle=0, rotation_point='center', facecolor='black')
    ax.add_patch(patch)

    plt.plot([-25, 25], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-25, -6], color='black', linestyle='--')
    plt.plot([0, 0], [25, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    ani = plot_vehicles(results, fig, ax, env, t_start, t_end)

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani, ax, fig

def plot_simulation_env_2(env, results, t_start, t_end, fig, ax):
    time = len(results['agent 0']['x coord'])
    fig, ax = plt.subplots()
    obstacles = env['Road Limits']
    for id in obstacles:
        """if obstacles[id]['radius'][0] != obstacles[id]['radius'][1]:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif obstacles[id]['radius'][0] == obstacles[id]['radius'][1]:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)"""
        ax.set_aspect('equal')
        plt.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    patch =  patches.Rectangle((-7, -6), 1, 6, angle=0 , rotation_point='center', facecolor='black')
    ax.add_patch(patch)
    patch = patches.Rectangle((6, 0), 1, 6, angle=0, rotation_point='center', facecolor='black')
    ax.add_patch(patch)

    plt.plot([-25, -6], [0, 0], color='black', linestyle='--')
    plt.plot([6, 25], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-25, 25], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    ani = plot_vehicles(results, fig, ax, env, t_start, t_end)

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani, ax, fig

def plot_simulation_env_3(env, results, t_start, t_end, fig, ax):
    time = len(results['agent 0']['x coord'])
    fig, ax = plt.subplots()
    obstacles = env['Road Limits']
    for id in obstacles:
        """if obstacles[id]['radius'][0] != obstacles[id]['radius'][1]:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif obstacles[id]['radius'][0] == obstacles[id]['radius'][1]:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)"""
        ax.set_aspect('equal')
        plt.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    plt.plot([-30, -6], [0, 0], color='black', linestyle='--')
    plt.plot([30, 6], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-30, -6], color='black', linestyle='--')
    plt.plot([0, 0], [30, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    ani = plot_vehicles(results, fig, ax, env, t_start, t_end)

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani, ax, fig

def plot_simulation_env_4(env, results, t_start, t_end, fig, ax):
    time = len(results['agent 0']['x coord'])
    fig, ax = plt.subplots()
    obstacles = env['Road Limits']
    for id in obstacles:
        """if obstacles[id]['radius'][0] != obstacles[id]['radius'][1]:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif obstacles[id]['radius'][0] == obstacles[id]['radius'][1]:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)"""
        ax.set_aspect('equal')
        plt.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    plt.plot([-30, -6], [0, 0], color='black', linestyle='--')
    plt.plot([30, 6], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-30, -6], color='black', linestyle='--')
    plt.plot([0, 0], [30, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    ani = plot_vehicles(results, fig, ax, env, t_start, t_end)

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani, ax, fig

def plot_simulation_env_5(env, results, t_start, t_end, fig, ax):
    time = len(results['agent 0']['x coord'])
    fig, ax = plt.subplots()
    obstacles = env['Road Limits']
    for id in obstacles:
        """if len(obstacles[id]['radius']) > 1:
            ellipse = patches.Ellipse((obstacles[id]['center'][0], obstacles[id]['center'][1]), (obstacles[id]['radius'][0]) * 2, (obstacles[id]['radius'][1]) * 2, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipse)
            ax.set_aspect('equal')
        elif len(obstacles[id]['radius']) == 1:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                      obstacles[id]['radius'][0], edgecolor='black',
                                      facecolor='black')"""
        if len(obstacles[id]['radius']) == 1:
            circle = patches.Circle((obstacles[id]['center'][0], obstacles[id]['center'][1]),
                                    obstacles[id]['radius'][0], edgecolor='black',
                                    facecolor='black')
            ax.set_aspect('equal')
            ax.add_patch(circle)
        ax.set_aspect('equal')
        plt.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    plt.plot([-25, -6], [0, 0], color='black', linestyle='--')
    plt.plot([25, 6], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-25, -6], color='black', linestyle='--')
    plt.plot([0, 0], [25, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')

    ani = plot_vehicles(results, fig, ax, env, t_start, t_end)

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

    return ani, ax, fig