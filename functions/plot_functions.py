import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

def plot_simulation(env_type, env, results):
    if env_type == 0:
        plot_simulation_env_0(env, results)
    elif env_type == 1:
        plot_simulation_env_1(env, results)
    elif env_type == 2:
        plot_simulation_env_2(env, results)
    elif env_type == 3:
        plot_simulation_env_3(env, results)
    elif env_type == 4:
        plot_simulation_env_4(env, results)
    elif env_type == 5:
        plot_simulation_env_5(env, results)
    else:
        print('Not ready')


def plot_simulation_env_0(env, results):
    time = len(results['agent 0']['x coord']) -1
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

    plt.plot([-25, -6], [0, 0], color='black', linestyle='--')
    plt.plot([25, 6], [0, 0], color='black', linestyle='--')
    plt.plot([0, 0], [-25, -6], color='black', linestyle='--')
    plt.plot([0, 0], [25, 6], color='black', linestyle='--')

    for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')


    vehicles = {}
    labels = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            if id_agent != len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle((results[f'agent {id_agent}']['x coord'][0] - L/2, results[f'agent 0']['y coord'][0]- W/2),
                                            L, W, angle=angle , rotation_point='center', facecolor='green', label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0], results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                ha='center', va='center', color='white')
            else:
                if env['With LLM car']:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='blue', label='LLM car')
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0], results[f'agent 0']['y coord'][0], 'LLM',
                                                    ha='center', va='center', color='black')
                else:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='green', label=str(id_agent))
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                    results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                    ha='center', va='center', color='white')
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)
            data = np.stack([results[f'agent {id_agent}']['x coord'][frame], results[f'agent {id_agent}']['y coord'][frame]]).T
            labels[f'{id_agent}'].set_position(data)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()


def plot_simulation_env_1(env, results):
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

    vehicles = {}
    labels = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            if id_agent != len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor='green', label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                ha='center', va='center', color='white')
            else:
                if env['With LLM car']:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='blue', label='LLM car')
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                    results[f'agent 0']['y coord'][0], 'LLM',
                                                    ha='center', va='center', color='black')
                else:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='green', label=str(id_agent))
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                    results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                    ha='center', va='center', color='white')
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)
            data = np.stack([results[f'agent {id_agent}']['x coord'][frame], results[f'agent {id_agent}']['y coord'][frame]]).T
            labels[f'{id_agent}'].set_position(data)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

def plot_simulation_env_2(env, results):
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

    vehicles = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            vehicles[f'{id_agent}'] = patches.Rectangle((results[f'agent {id_agent}']['x coord'][0] - L/2, results[f'agent 0']['y coord'][0]- W/2),
                                            L, W, angle=angle , rotation_point='center', facecolor='green')
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

def plot_simulation_env_3(env, results):
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


    vehicles = {}
    labels = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            if id_agent != len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor='green', label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                ha='center', va='center', color='white')
            else:
                if env['With LLM car']:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='blue', label='LLM car')
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                    results[f'agent 0']['y coord'][0], 'LLM',
                                                    ha='center', va='center', color='black')
                else:
                    vehicles[f'{id_agent}'] = patches.Rectangle(
                        (results[f'agent {id_agent}']['x coord'][0] - L / 2, results[f'agent 0']['y coord'][0] - W / 2),
                        L, W, angle=angle, rotation_point='center', facecolor='green', label=str(id_agent))
                    labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][0],
                                                    results[f'agent 0']['y coord'][0], f'{id_agent}',
                                                    ha='center', va='center', color='white')
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)
            data = np.stack([results[f'agent {id_agent}']['x coord'][frame], results[f'agent {id_agent}']['y coord'][frame]]).T
            labels[f'{id_agent}'].set_position(data)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

def plot_simulation_env_4(env, results):
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


    vehicles = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            if results[f'agent {id_agent}']['type'] == 'standard_car':
                color = 'green'
            elif results[f'agent {id_agent}']['type'] == 'emergency_car':
                color = 'red'
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            vehicles[f'{id_agent}'] = patches.Rectangle((results[f'agent {id_agent}']['x coord'][0] - L/2, results[f'agent 0']['y coord'][0]- W/2),
                                            L, W, angle=angle , rotation_point='center', facecolor=color)
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()

def plot_simulation_env_5(env, results):
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


    vehicles = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            if results[f'agent {id_agent}']['type'] == 'standard_car':
                color = 'green'
            elif results[f'agent {id_agent}']['type'] == 'emergency_car':
                color = 'red'
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][0] * 180 / np.pi
            vehicles[f'{id_agent}'] = patches.Rectangle((results[f'agent {id_agent}']['x coord'][0] - L/2, results[f'agent 0']['y coord'][0]- W/2),
                                            L, W, angle=angle , rotation_point='center', facecolor=color)
            ax.add_patch(vehicles[f'{id_agent}'])

    """lines = {}
    scats = {}
    for k in range(len(results)):
        lines[f'line{k}'] = ax.plot(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="b")[0]  # label=f'v0 = {v02} m/s'
        scats[f'line{k}'] = ax.scatter(results[f'agent {k}']['x coord'][0], results[f'agent {k}']['y coord'][0], c="r", s=5, label=f'car_{k + 1}')
    """

    def update(frame):

        for id_agent in vehicles:
            x = results[f'agent {id_agent}']['x coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']/2
            y = results[f'agent {id_agent}']['y coord'][frame] - env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']/2
            angle = results[f'agent {id_agent}']['theta'][frame] * 180 / np.pi
            data = np.stack([x, y]).T
            vehicles[f'{id_agent}'].set_xy(data)
            vehicles[f'{id_agent}'].set_angle(angle)

        """for k in range(len(results)):
            x = results[f'agent {k}']['x coord'][:frame]
            y = results[f'agent {k}']['y coord'][:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scats[f'line{k}'].set_offsets(data)
            # update the line plot:
            lines[f'line{k}'].set_xdata(results[f'agent {k}']['x coord'][:frame])
            lines[f'line{k}'].set_ydata(results[f'agent {k}']['y coord'][:frame])"""

        return (vehicles)

    ani = FuncAnimation(fig=fig, func=update, frames=time, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.show()