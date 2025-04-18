import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

def save_all_frames(results, env):
    for t in range(len(results['agent 0']['x coord'])):
        fig, ax = plt.subplots()
        if env['env number'] == 0:
            ani, ax, fig = plot_simulation_env_0(env, results, None, None, fig, ax)
        elif env['env number'] == 1:
            ani, ax, fig = plot_simulation_env_1(env, results, None, None, fig, ax)
        elif env['env number'] == 2:
            ani, ax, fig = plot_simulation_env_2(env, results, None, None, fig, ax)
        elif env['env number'] == 3:
            ani, ax, fig = plot_simulation_env_3(env, results, None, None, fig, ax)
        elif env['env number'] == 4:
            ani, ax, fig = plot_simulation_env_4(env, results, None, None, fig, ax)
        elif env['env number'] == 5:
            ani, ax, fig = plot_simulation_env_5(env, results, None, None, fig, ax)

        ax.set_aspect('equal')

        vehicles, labels, lines, ellipses, hulls, ax = prep_plot_vehicles(results, env, t, ax)

        ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
        ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

        path_fig = os.path.join(os.path.dirname(__file__), "..", f"save_results/images_sim/")
        fig.savefig(path_fig + f'{t}.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        acc_SF, acc_LLM, line1, line2, ax = prep_plot_acc_input(results, t, ax)

        ax.set_xlim(0 - 0.5,len(results['agent 0']['x coord'])  + 0.5)
        ax.set_ylim(min(min(acc_SF), min(acc_LLM)) - 0.5, max(max(acc_SF), max(acc_LLM)) + 0.5)

        path_fig = os.path.join(os.path.dirname(__file__), "..", f"save_results/images_acc/")
        fig.savefig(path_fig + f'{t}.png')
        plt.close(fig)

        fig, ax = plt.subplots()

        steering_SF, steering_LLM, line3, line4, ax = prep_plot_steer_input(results, t, ax)

        ax.set_xlim(0 - 0.5, len(results['agent 0']['x coord']) + 0.5)
        ax.set_ylim(min(min(steering_SF), min(steering_LLM)) - 0.5, max(max(steering_SF), max(steering_LLM)) + 0.5)

        path_fig = os.path.join(os.path.dirname(__file__), "..", f"save_results/images_steer/")
        fig.savefig(path_fig + f'{t}.png')
        plt.close(fig)

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

    #plt.show()


def input_animation(results, t_start, t_end):
    fig, ax1= plt.subplots()

    """time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    acc_SF = results[f'agent {str(len(results)-1)}']['acc pred SF']
    acc_LLM = results[f'agent {str(len(results) - 1)}']['acc pred LLM']

    line1, = ax1.plot(time[t_start], acc_SF[t_start], color='orange')
    line2, = ax1.plot(time[t_start], acc_LLM[t_start], color='red')

    ax1.legend(['SF', 'LLM'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('[m/s^2]')
    ax1.set_title('Acceleration')"""

    acc_SF, acc_LLM, line1, line2, ax1 = prep_plot_acc_input(results, t_start, ax1)

    time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    ax1.set_xlim(min(time[t_start:t_end]) - 0.5, max(time[t_start:t_end]) + 0.5)
    ax1.set_ylim(min(min(acc_SF), min(acc_LLM)) - 0.5, max(max(acc_SF), max(acc_LLM)) + 0.5)

    def update1(frame):
        line1.set_xdata(time[t_start:t_start+frame])
        line1.set_ydata(acc_SF[t_start:t_start+frame])
        line2.set_xdata(time[t_start:t_start+frame])
        line2.set_ydata(acc_LLM[t_start:t_start+frame])
        return line1, line2

    # Create animations
    ani1 = FuncAnimation(fig=fig, func=update1, frames=t_end-t_start, interval=50, blit=True, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani1.save(path + '/animation/acc_input.gif', writer='pillow')
    #plt.show()
    plt.close(fig)

    fig, ax2 = plt.subplots()

    """time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    steering_SF = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred SF']]
    steering_LLM = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred LLM']]

    line3, = ax2.plot(time[t_start], steering_SF[t_start], color='orange')
    line4, = ax2.plot(time[t_start], steering_LLM[t_start], color='red')

    ax2.legend(['SF', 'LLM'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('[deg]')
    ax2.set_title('Steering angle')"""

    steering_SF, steering_LLM, line3, line4, ax2 = prep_plot_steer_input(results, t_start, ax2)

    ax2.set_xlim(min(time[t_start:t_end]) - 0.5, max(time[t_start:t_end]) + 0.5)
    ax2.set_ylim(min(min(steering_SF), min(steering_LLM)) - 0.5, max(max(steering_SF), max(steering_LLM)) + 0.5)

    def update2(frame):
        line3.set_xdata(time[t_start:t_start+frame])
        line3.set_ydata(steering_SF[t_start:t_start+frame])
        line4.set_xdata(time[t_start:t_start+frame])
        line4.set_ydata(steering_LLM[t_start:t_start+frame])

        return line3, line4

    ani2 = FuncAnimation(fig=fig, func=update2, frames=t_end-t_start, interval=50, blit=True, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani2.save(path + '/animation/steering_input.gif', writer='pillow')

    #plt.show()
    plt.close(fig)

    return ani1, ani2


def prep_plot_steer_input(results, t_start, ax2):

    time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    steering_SF = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred SF']]
    steering_LLM = [x * 180 / np.pi for x in results[f'agent {str(len(results) - 1)}']['steering angle pred LLM']]

    line3, = ax2.plot(time[:t_start], steering_SF[:t_start], color='orange')
    line4, = ax2.plot(time[:t_start], steering_LLM[:t_start], color='red')

    ax2.legend(['SF', 'LLM'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('[deg]')
    ax2.set_title('Steering angle')

    return steering_SF, steering_LLM, line3, line4, ax2

def prep_plot_acc_input(results, t_start, ax1):

    time = np.arange(len(results[f'agent {str(len(results) - 1)}']['acc pred SF']))
    acc_SF = results[f'agent {str(len(results) - 1)}']['acc pred SF']
    acc_LLM = results[f'agent {str(len(results) - 1)}']['acc pred LLM']

    line1, = ax1.plot(time[:t_start], acc_SF[:t_start], color='orange')
    line2, = ax1.plot(time[:t_start], acc_LLM[:t_start], color='red')

    ax1.legend(['SF', 'LLM'])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('[m/s^2]')
    ax1.set_title('Acceleration')

    return acc_SF, acc_LLM, line1, line2, ax1

def prep_plot_vehicles(results, env, t_start, ax):

    vehicles = {}
    labels = {}
    ellipses = {}
    hulls = {}
    for id_agent in range(len(results)):
        if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
            L = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][t_start] * 180 / np.pi
            if results[f'agent {id_agent}']['type'] == 'standard car':
                color = 'green'
            elif results[f'agent {id_agent}']['type'] == 'emergency car':
                color = 'red'
            if env['With LLM car'] and id_agent == len(results) - 1:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                     results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor='blue', label='EGO car')
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                results[f'agent {id_agent}']['y coord'][t_start], 'EGO',
                                                ha='center', va='center', color='black')
                ax.add_patch(vehicles[f'{id_agent}'])
                if results[f'agent {id_agent}']['safe set'][t_start][4]:
                    ellipse_color = 'black'
                else:
                    ellipse_color = 'red'
                ellipses[f'{id_agent}'] = patches.Circle((results[f'agent {id_agent}']['safe set'][t_start][1][0],
                                                           results[f'agent {id_agent}']['safe set'][t_start][1][1]),
                                                          results[f'agent {id_agent}']['safe set'][t_start][0],
                                                          edgecolor=ellipse_color, linestyle='--', linewidth=1, facecolor='none')
                ax.add_patch(ellipses[f'{id_agent}'])

                hulls[f'{id_agent}'] = patches.Polygon(np.array(results[f'agent {id_agent}']['hull'][t_start]),
                                                       closed=True, fill=False, color='blue', alpha=0.5)
                ax.add_patch(hulls[f'{id_agent}'])

            else:
                vehicles[f'{id_agent}'] = patches.Rectangle(
                    (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                     results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                    L, W, angle=angle, rotation_point='center', facecolor=color, label=str(id_agent))
                labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                                results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                                ha='center', va='center', color='black')
                ax.add_patch(vehicles[f'{id_agent}'])

                """ellipses[f'{id_agent}'] = patches.Ellipse((results[f'agent {id_agent}']['x coord'][t_start],
                                                           results[f'agent {id_agent}']['y coord'][t_start]), 8, 8,
                                                          angle=angle, edgecolor='black', linestyle='--', linewidth=1,
                                                          facecolor='none')
                ax.add_patch(ellipses[f'{id_agent}'])"""

                hulls[f'{id_agent}'] = patches.Polygon(np.array(results[f'agent {id_agent}']['hull'][t_start]),
                                                       closed=True, fill=False, color='blue', alpha=0.5)
                ax.add_patch(hulls[f'{id_agent}'])

        elif results[f'agent {id_agent}']['type'] in env['Pedestrians Specification']['types']:
            L = env['Pedestrians Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Pedestrians Specification'][results[f'agent {id_agent}']['type']]['width']
            vehicles[f'{id_agent}'] = patches.Rectangle(
                (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                 results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                L, W, angle=0, facecolor='magenta', label=str(id_agent))
            labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                            results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                            ha='center', va='center', color='black')
            ax.add_patch(vehicles[f'{id_agent}'])
            """ellipses[f'{id_agent}'] = patches.Ellipse((results[f'agent {id_agent}']['x coord'][t_start],
                                                       results[f'agent {id_agent}']['y coord'][t_start]), 8, 8,
                                                      edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipses[f'{id_agent}'])"""
            hulls[f'{id_agent}'] = patches.Polygon(np.array(results[f'agent {id_agent}']['hull'][t_start]),
                                                   closed=True, fill=False, color='blue', alpha=0.5)
            ax.add_patch(hulls[f'{id_agent}'])
        elif results[f'agent {id_agent}']['type'] in env['Bicycle Specification']['types']:
            L = env['Bicycle Specification'][results[f'agent {id_agent}']['type']]['length']
            W = env['Bicycle Specification'][results[f'agent {id_agent}']['type']]['width']
            angle = results[f'agent {id_agent}']['theta'][t_start] * 180 / np.pi
            vehicles[f'{id_agent}'] = patches.Rectangle(
                (results[f'agent {id_agent}']['x coord'][t_start] - L / 2,
                 results[f'agent {id_agent}']['y coord'][t_start] - W / 2),
                L, W, angle=angle, rotation_point='center', facecolor='yellow', label=str(id_agent))
            labels[f'{id_agent}'] = ax.text(results[f'agent {id_agent}']['x coord'][t_start],
                                            results[f'agent {id_agent}']['y coord'][t_start], f'{id_agent}',
                                            ha='center', va='center', color='black')
            ax.add_patch(vehicles[f'{id_agent}'])
            """ellipses[f'{id_agent}'] = patches.Ellipse((results[f'agent {id_agent}']['x coord'][t_start],
                                                       results[f'agent {id_agent}']['y coord'][t_start]), 8, 8,
                                                      edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
            ax.add_patch(ellipses[f'{id_agent}'])"""
            hulls[f'{id_agent}'] = patches.Polygon(np.array(results[f'agent {id_agent}']['hull'][t_start]),
                                                   closed=True, fill=False, color='blue', alpha=0.5)
            ax.add_patch(hulls[f'{id_agent}'])

    lines = {}
    # scats = {}
    for k in range(len(results)):
        if env['With LLM car'] and k == len(results) - 1:
            lines[f'line{k} SF'] = \
                ax.plot(results[f'agent {k}']['x coord pred SF'][t_start],
                        results[f'agent {k}']['y coord pred SF'][t_start],
                        c="orange", linestyle='-')[0]  # label=f'v0 = {v02} m/s'
            lines[f'line{k} traj estimation'] = \
                ax.plot(results[f'agent {k}']['safe set'][t_start][2],
                        results[f'agent {k}']['safe set'][t_start][3], c="green", linestyle='-')[0]
            lines[f'line{k}'] = \
                ax.plot(results[f'agent {k}']['x coord pred'][t_start],
                        results[f'agent {k}']['y coord pred'][t_start],
                        c="red", linestyle='-')[0]  # label=f'v0 = {v02} m/s'
        """else:
            lines[f'line{k} traj estimation'] = \
                ax.plot(results[f'agent {k}']['trajectory estimation x'][t_start],
                        results[f'agent {k}']['trajectory estimation y'][t_start], c="green", linestyle='-')[0]  # label=f'v0 = {v02} m/s'
            if results[f'agent {k}']['type'] in env['Vehicle Specification']['types']:
                lines[f'line{k}'] = \
                    ax.plot(results[f'agent {k}']['x coord pred'][t_start],
                            results[f'agent {k}']['y coord pred'][t_start],
                            c="red", linestyle='-')[0]  # label=f'v0 = {v02} m/s'"""

    return vehicles, labels, lines, ellipses, hulls, ax

def plot_vehicles(results, fig, ax, env, t_start, t_end):

    vehicles, labels, lines, ellipses, hulls, ax = prep_plot_vehicles(results, env, t_start, ax)
    def update(frame):

        for id_agent in vehicles:
            if results[f'agent {id_agent}']['type'] in env['Vehicle Specification']['types']:
                # Rectangle
                x = results[f'agent {id_agent}']['x coord'][t_start+frame] - \
                    env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['length'] / 2
                y = results[f'agent {id_agent}']['y coord'][t_start+frame] - \
                    env['Vehicle Specification'][results[f'agent {id_agent}']['type']]['width'] / 2
                angle = results[f'agent {id_agent}']['theta'][t_start+frame] * 180 / np.pi
                data = np.stack([x, y]).T
                vehicles[f'{id_agent}'].set_xy(data)
                vehicles[f'{id_agent}'].set_angle(angle)
                #Labels of Rectangle
                data = np.stack(
                    [results[f'agent {id_agent}']['x coord'][t_start+frame], results[f'agent {id_agent}']['y coord'][t_start+frame]]).T
                labels[f'{id_agent}'].set_position(data)
                #Trajectories predicted
                if env['With LLM car'] and id_agent == str(len(results) - 1):
                    lines[f'line{id_agent} SF'].set_xdata(results[f'agent {id_agent}']['x coord pred SF'][t_start+frame])
                    lines[f'line{id_agent} SF'].set_ydata(results[f'agent {id_agent}']['y coord pred SF'][t_start+frame])
                    lines[f'line{id_agent} traj estimation'].set_xdata(results[f'agent {id_agent}']['safe set'][t_start + frame][2])
                    lines[f'line{id_agent} traj estimation'].set_ydata(results[f'agent {id_agent}']['safe set'][t_start + frame][3])
                    lines[f'line{id_agent}'].set_xdata(results[f'agent {id_agent}']['x coord pred'][t_start + frame])
                    lines[f'line{id_agent}'].set_ydata(results[f'agent {id_agent}']['y coord pred'][t_start + frame])
                    # Safe set -> ellipsoid
                    x = results[f'agent {id_agent}']['safe set'][t_start + frame][1][0]
                    y = results[f'agent {id_agent}']['safe set'][t_start + frame][1][1]
                    R = results[f'agent {id_agent}']['safe set'][t_start + frame][0]
                    ellipses[f'{id_agent}'].center = (x, y)
                    ellipses[f'{id_agent}'].radius = R
                    ellipses[f'{id_agent}'].set_edgecolor('black')
                    # Convex Hulls
                    hull_frame = np.array(results[f'agent {id_agent}']['hull'][t_start + frame])
                    hulls[f'{id_agent}'].set_xy(hull_frame)
                    if not results[f'agent {id_agent}']['safe set'][t_start + frame][4]:
                        ellipses[f'{id_agent}'].set_edgecolor('white')
                        hulls[f'{id_agent}'].set_edgecolor('white')
                        lines[f'line{id_agent} SF'].set_color('white')
                        lines[f'line{id_agent} traj estimation'].set_color('white')
                        lines[f'line{id_agent}'].set_color('white')
                    elif not results[f'agent {id_agent}']['safe set'][t_start + frame][5]:
                        ellipses[f'{id_agent}'].set_edgecolor('red')
                        hulls[f'{id_agent}'].set_edgecolor('blue')
                        lines[f'line{id_agent} SF'].set_color('orange')
                        lines[f'line{id_agent} traj estimation'].set_color('green')
                        lines[f'line{id_agent}'].set_color('red')
                    else:
                        ellipses[f'{id_agent}'].set_edgecolor('black')
                        hulls[f'{id_agent}'].set_edgecolor('blue')
                        lines[f'line{id_agent} SF'].set_color('orange')
                        lines[f'line{id_agent} traj estimation'].set_color('green')
                        lines[f'line{id_agent}'].set_color('red')
                else:
                    # Print the trajecotries
                    #lines[f'line{id_agent} traj estimation'].set_xdata(
                    #    results[f'agent {id_agent}']['trajectory estimation x'][t_start+frame])
                    #lines[f'line{id_agent} traj estimation'].set_ydata(
                    #    results[f'agent {id_agent}']['trajectory estimation y'][t_start+frame])
                    #lines[f'line{id_agent}'].set_xdata(results[f'agent {id_agent}']['x coord pred'][t_start + frame])
                    #lines[f'line{id_agent}'].set_ydata(results[f'agent {id_agent}']['y coord pred'][t_start + frame])

                    # Security areas -> ellipsoid
                    x = results[f'agent {id_agent}']['x coord'][t_start + frame]
                    y = results[f'agent {id_agent}']['y coord'][t_start + frame]
                    angle = results[f'agent {id_agent}']['theta'][t_start + frame] * 180 / np.pi
                    #ellipses[f'{id_agent}'].center = (x, y)
                    #ellipses[f'{id_agent}'].angle = angle

                    # Convex Hulls
                    hull_frame = np.array(results[f'agent {id_agent}']['hull'][t_start + frame])
                    hulls[f'{id_agent}'].set_xy(hull_frame)

            elif results[f'agent {id_agent}']['type'] in env['Pedestrians Specification']['types']:
                hull_frame = np.array(results[f'agent {id_agent}']['hull'][t_start + frame])
                hulls[f'{id_agent}'].set_xy(hull_frame)
                # Rectangle
                x = results[f'agent {id_agent}']['x coord'][t_start + frame] - \
                    env['Pedestrians Specification'][results[f'agent {id_agent}']['type']]['length'] / 2
                y = results[f'agent {id_agent}']['y coord'][t_start + frame] - \
                    env['Pedestrians Specification'][results[f'agent {id_agent}']['type']]['width'] / 2
                data = np.stack([x, y]).T
                vehicles[f'{id_agent}'].set_xy(data)
                data = np.stack(
                    [results[f'agent {id_agent}']['x coord'][t_start + frame],
                     results[f'agent {id_agent}']['y coord'][t_start + frame]]).T
                labels[f'{id_agent}'].set_position(data)
                #lines[f'line{id_agent} traj estimation'].set_xdata(
                #    results[f'agent {id_agent}']['trajectory estimation x'][t_start + frame])
                #lines[f'line{id_agent} traj estimation'].set_ydata(
                #    results[f'agent {id_agent}']['trajectory estimation y'][t_start + frame])
                x = results[f'agent {id_agent}']['x coord'][t_start + frame]
                y = results[f'agent {id_agent}']['y coord'][t_start + frame]
                #ellipses[f'{id_agent}'].center = (x, y)
            elif results[f'agent {id_agent}']['type'] in env['Bicycle Specification']['types']:
                hull_frame = np.array(results[f'agent {id_agent}']['hull'][t_start + frame])
                hulls[f'{id_agent}'].set_xy(hull_frame)
                # Rectangle
                x = results[f'agent {id_agent}']['x coord'][t_start+frame] - \
                    env['Bicycle Specification'][results[f'agent {id_agent}']['type']]['length'] / 2
                y = results[f'agent {id_agent}']['y coord'][t_start+frame] - \
                    env['Bicycle Specification'][results[f'agent {id_agent}']['type']]['width'] / 2
                angle = results[f'agent {id_agent}']['theta'][t_start+frame] * 180 / np.pi
                data = np.stack([x, y]).T
                vehicles[f'{id_agent}'].set_xy(data)
                vehicles[f'{id_agent}'].set_angle(angle)
                #Labels of Rectangle
                data = np.stack(
                    [results[f'agent {id_agent}']['x coord'][t_start+frame], results[f'agent {id_agent}']['y coord'][t_start+frame]]).T
                labels[f'{id_agent}'].set_position(data)
                #Trajectories predicted
                #lines[f'line{id_agent} traj estimation'].set_xdata(
                #    results[f'agent {id_agent}']['trajectory estimation x'][t_start+frame])
                #lines[f'line{id_agent} traj estimation'].set_ydata(
                #    results[f'agent {id_agent}']['trajectory estimation y'][t_start+frame])
                #lines[f'line{id_agent}'].set_xdata(results[f'agent {id_agent}']['x coord pred'][t_start+frame])
                #lines[f'line{id_agent}'].set_ydata(results[f'agent {id_agent}']['y coord pred'][t_start+frame])
                #Security areas -> ellipsoid
                shift = np.array([[np.cos(results[f'agent {id_agent}']['theta'][t_start + frame]) * 1.5],
                                  [np.sin(results[f'agent {id_agent}']['theta'][t_start + frame]) * 1.5]])
                x = results[f'agent {id_agent}']['x coord'][t_start + frame]
                y = results[f'agent {id_agent}']['y coord'][t_start + frame]
                angle = results[f'agent {id_agent}']['theta'][t_start + frame] * 180 / np.pi
                #ellipses[f'{id_agent}'].center = (x, y)
                #ellipses[f'{id_agent}'].angle = angle

        return (vehicles, labels, lines, ellipses)

    ani = FuncAnimation(fig=fig, func=update, frames=t_end-t_start, interval=50, repeat=False)

    path = os.path.join(os.path.dirname(__file__), "..")
    ani.save(path + '/animation/animation.gif', writer='pillow')

    return ani

def plot_simulation(env_type, env, results, t_start, t_end):

    fig, ax = plt.subplots()

    if env_type == 0 or env_type == 6:
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

    #plt.show()
    plt.close(fig)

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

    #for id in env['Entrances']:
    #    ax.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
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
        ax.plot(obstacles[id]['line x'], obstacles[id]['line y'], color='black')

    patch =  patches.Rectangle((0, -7), 6, 1, angle=0 , rotation_point='center', facecolor='black')
    ax.add_patch(patch)
    patch = patches.Rectangle((-6, 6), 6, 1, angle=0, rotation_point='center', facecolor='black')
    ax.add_patch(patch)

    ax.plot([-25, 25], [0, 0], color='black', linestyle='--')
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

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig

def plot_simulation_env_2(env, results, t_start, t_end, fig, ax):

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

    if t_start != None:
        ani = plot_vehicles(results, fig, ax, env, t_start, t_end)
    else:
        ani = None

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig

def plot_simulation_env_3(env, results, t_start, t_end, fig, ax):

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

    if t_start != None:
        ani = plot_vehicles(results, fig, ax, env, t_start, t_end)
    else:
        ani = None

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig

def plot_simulation_env_4(env, results, t_start, t_end, fig, ax):

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

    """for id in env['Entrances']:
        plt.scatter(env['Entrances'][id]['position'][0], env['Entrances'][id]['position'][1], color='magenta')
    for id in env['Exits']:
        plt.scatter(env['Exits'][id]['position'][0], env['Exits'][id]['position'][1], color='blue')"""

    if t_start != None:
        ani = plot_vehicles(results, fig, ax, env, t_start, t_end)
    else:
        ani = None

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig

def plot_simulation_env_5(env, results, t_start, t_end, fig, ax):

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

    if t_start != None:
        ani = plot_vehicles(results, fig, ax, env, t_start, t_end)
    else:
        ani = None

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    #plt.show()

    return ani, ax, fig

def plot_frame_for_describer(env_type, env, agents, ego, t):
    fig, ax = plt.subplots()

    if env_type == 0:
        ani, ax, fig = plot_simulation_env_0(env, None, None, None, fig, ax)
    elif env_type == 1:
        ani, ax, fig = plot_simulation_env_1(env, None, None, None, fig, ax)
    elif env_type == 2:
        ani, ax, fig = plot_simulation_env_2(env, None, None, None, fig, ax)
    elif env_type == 3:
        ani, ax, fig = plot_simulation_env_3(env, None, None, None, fig, ax)
    elif env_type == 4:
        ani, ax, fig = plot_simulation_env_4(env, None, None, None, fig, ax)
    elif env_type == 5:
        ani, ax, fig = plot_simulation_env_5(env, None, None, None, fig, ax)
    else:
        print('Not ready')

    for name_agent in agents:
        if agents[name_agent].type in env['Vehicle Specification']['types']:
            if np.linalg.norm(agents[name_agent].position - ego.final_target['position']) > 1:
                L = agents[name_agent].length
                W = agents[name_agent].width
                angle = agents[name_agent].theta * 180 / np.pi
                if agents[name_agent].type == 'standard car':
                    color = 'green'
                elif agents[name_agent].type == 'emergency car':
                    color = 'red'
                rectangle = patches.Rectangle((agents[name_agent].x - L / 2, agents[name_agent].y - W / 2), L, W,
                                              angle=angle, rotation_point='center', facecolor=color, label=str(name_agent))
                ax.text(agents[name_agent].x, agents[name_agent].y, f'{name_agent}', ha='center', va='center', color='white')
                ax.add_patch(rectangle)

                shift = np.array([[np.cos(agents[name_agent].theta) * 1.5],
                                  [np.sin(agents[name_agent].theta) * 1.5]])

                ellipse = patches.Ellipse((agents[name_agent].x, agents[name_agent].y),
                                          agents[name_agent].a_security_dist * 2, agents[name_agent].b_security_dist * 2,
                                          angle=angle, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
                ax.add_patch(ellipse)

                agents[name_agent].trajecotry_estimation()
                ax.plot(agents[name_agent].traj_estimation[0, :], agents[name_agent].traj_estimation[1, :], c="green", linestyle='-')

    if env['With LLM car']:
        angle = ego.theta * 180 / np.pi
        rectangle = patches.Rectangle((ego.x - L / 2, ego.y - W / 2), L, W, angle=angle, rotation_point='center',
                                      facecolor='blue', label='LLM car')
        ax.text(ego.x, ego.y, 'LLM', ha='center', va='center', color='black')

        ax.add_patch(rectangle)

        shift = np.array([[np.cos(ego.theta) * 1.5],
                          [np.sin(ego.theta) * 1.5]])

        ellipse = patches.Ellipse((ego.x, ego.y),agents[name_agent].a_security_dist * 2, agents[name_agent].b_security_dist * 2,
                                  angle=angle, edgecolor='black', linestyle='--', linewidth=1, facecolor='none')
        ax.add_patch(ellipse)

        ego.trajecotry_estimation()
        ax.plot(ego.traj_estimation[0, :], ego.traj_estimation[1, :], c="green", linestyle='-')

    ax.set_xlim(env["State space"]["x limits"][0], env["State space"]["x limits"][1])
    ax.set_ylim(env["State space"]["y limits"][0], env["State space"]["y limits"][1])

    plt.scatter(ego.entry['x'],  ego.entry['y'], color='blue')
    plt.text(ego.entry['x'] - 2 ,  ego.entry['y'] + 1.1, 'entry', fontsize=10, color='blue')
    plt.scatter(ego.exit['x'], ego.exit['y'], color='blue')
    plt.text(ego.exit['x'] - 2, ego.exit['y'] + 1.1, 'exit', fontsize=10, color='blue')
    plt.scatter(ego.final_target['x'], ego.final_target['y'], color='blue')
    plt.text(ego.final_target['x'] - 5.1, ego.final_target['y']+ 1.1, 'final_target', fontsize=10, color='blue')

    path_fig = os.path.join(os.path.dirname(__file__), "..", f"prompts/output_LLM/frames/")
    fig.savefig(path_fig + f'frame_{t}.png')
    plt.close(fig)

    return fig