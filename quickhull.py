import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_points():
    """
    Generate random points (x,y)
    :return: List of randomized point
    :rtype: DataFrame
    """
    try:
        dist = input("Enter 'R' for random points distribution or 'C' for circular distribution: ").lower()
    except:
        dist = 'r'

    if dist == 'c':
        return circle_points()
    else:
        try:
            count = int(input("Enter number of points: "))
        except:
            count = 200
        return random_distribution(count)


def random_distribution(count):
    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1

    x_list = np.random.uniform(x_min, x_max, size=count)
    y_list = np.random.uniform(y_min, y_max, size=count)

    point_list = pd.DataFrame({'x': x_list, 'y': y_list})
    print(point_list)
    return point_list


def circ_fig():
    n = [1, 10, 20, 30, 40, 50, 60]
    r = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    concentric_circles = []

    for r, n in zip(r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        concentric_circles.append(np.c_[x, y])

    fig, ax = plt.subplots()
    for circle in concentric_circles:
        ax.scatter(circle[:, 0], circle[:, 1])
    ax.set_aspect('equal')
    plt.show()


def circle_points():
    T = [1, 10, 20, 30, 40, 50, 60]
    R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    x_list = []
    y_list = []

    fig, ax = plt.subplots()
    for r, t in rtpairs(R, T):
        x_list.append(r * np.cos(t))
        y_list.append(r * np.sin(t))
        ax.plot(r * np.cos(t), r * np.sin(t), 'o')
    ax.set_aspect('equal')
    plt.show()

    point_list = pd.DataFrame({'x': x_list, 'y': y_list})
    print(point_list)
    return point_list


def rtpairs(r, n):
    for i in range(len(r)):
        for j in range(n[i]):
            yield r[i], j*(2 * np.pi / n[i])


def check_cond(point_1, point_2, check_point):
    """
    Check a point position based on a line created by 2 points (above / below / on)
    :param point_1: line first point
    :type point_1: DataFrame element
    :param point_2: line second point
    :type point_2: DataFrame element
    :param check_point: point which is going to be checked
    :type check_point: DataFrame element
    :return: point position
    :rtype: string
    """

    try:
        p1_x, p1_y, p1_c = point_1
    except:
        p1_x, p1_y = point_1

    try:
        p2_x, p2_y, p2_c = point_2
    except:
        p2_x, p2_y = point_2

    try:
        pc_x, pc_y, pc_c = check_point
    except:
        pc_x, pc_y = check_point

    # Use determinant to determine position
    det = (p2_x - p1_x) * (pc_y - p1_y) - (p2_y - p1_y) * (pc_x - p1_x)

    if det > 0:
        return "above"
    elif det < 0:
        return "below"
    else:
        return "on"


def filter_above_below(left_most, right_most, point_list):
    """
    Get all above/below condition
    :param left_most: line left most point (end)
    :type left_most: DataFrame element
    :param right_most: line right most point (end)
    :type right_most: DataFrame element
    :param point_list: list of points
    :type point_list: DataFrame
    :return: list of points on line, above and below
    :rtype: DataFrame
    """

    # Create an empty list for handle point position
    temp = np.empty(point_list.index[-1] + 1, dtype='U8')

    # Check every point position
    for index, row in point_list.iterrows():
        temp[index] = check_cond(left_most, right_most, row)

    # Filter out None in list
    temp = temp[temp != ''].copy()

    # Add a column of condition (above / below)
    point_list['cond'] = temp

    # Return copy of masked data frame
    line_above = point_list.loc[point_list['cond'] == 'above'].copy()
    line_below = point_list.loc[point_list['cond'] == 'below'].copy()

    return line_above, line_below


def left_most_point(point_list):
    """
    Get the left most point
    :param point_list: list of point
    :type point_list: DataFrame
    :return: left most point
    :rtype: DataFrame element
    """
    return point_list.loc[point_list['x'].idxmin()]


def right_most_point(point_list):
    """
    Get the right most point
    :param point_list: list of point
    :type point_list: DataFrame
    :return: right most point
    :rtype: DataFrame element
    """
    return point_list.loc[point_list['x'].idxmax()]


def get_furthest_point(line_above, line_below, point_list):
    """
    Get furthest point
    :param line_above: line above point
    :type line_above: DataFrame element
    :param line_below: line below point
    :type line_below: DataFrame element
    :param point_list: list of point to check
    :type point_list: DataFrame
    :return: furthest point
    :rtype: DataFrame element
    """

    def distance(line_above, line_below, point):
        """
        Get distance from a point to the line created by 2 points
        :param line_above: line above point
        :type line_above: DataFrame element
        :param line_below: line below point
        :type line_below: DataFrame element
        :param point: point to check
        :type point: DataFrame element
        :return: distance
        :rtype: float
        """

        try:
            x0, y0, c0 = point
        except:
            x0, y0 = point

        try:
            x1, y1, c1 = line_above
        except:
            x1, y1 = line_above

        try:
            x2, y2, c2 = line_below
        except:
            x2, y2 = line_below

        # Calculate distance
        top_eq = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        bot_eq = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

        return top_eq / bot_eq

    furthest_point = None
    max_distance = 0

    # Find the furthest
    for index, point in point_list.iterrows():
        point_distance = distance(line_above, line_below, point)
        if point_distance > max_distance:
            max_distance = point_distance
            furthest_point = point

    return furthest_point


def quick_hull(line_above, line_below, point_list):
    """
    Algorithm to find the most outer line of a clustered points
    :param line_above: line above edge
    :type line_above: DataFrame element
    :param line_below: line below edge
    :type line_below: DataFrame element
    :param point_list: list of point to check
    :type point_list: DataFrame
    :return: ines connecting the outer points
    :rtype: list
    """

    # If there's no more point to check (basis)
    if len(point_list['x']) == 0:
        return [(line_above, line_below)]

    # Get furthest point from line
    furthest_point = get_furthest_point(line_above, line_below, point_list)

    # Divide point into ABOVE-line's above / below
    point_on_line_above, point_on_line_below = filter_above_below(line_above, furthest_point, point_list)

    # Delete side that's useless
    delete_side = check_cond(line_above, furthest_point, line_below)

    # Choose one side that's outer
    rec_point_list = None
    if delete_side == "below":
        rec_point_list = point_on_line_above
    elif delete_side == "above":
        rec_point_list = point_on_line_below

    # Recurse
    outer_line_above = quick_hull(line_above, furthest_point, rec_point_list)

    # Divide point into BELOW-line's above / below
    point_on_line_above, point_on_line_below = filter_above_below(furthest_point, line_below, point_list)

    # Delete side that's useless
    delete_side = check_cond(furthest_point, line_below, line_above)

    # Choose one side that's outer
    rec_point_list = None
    if delete_side == "below":
        rec_point_list = point_on_line_above
    elif delete_side == "above":
        rec_point_list = point_on_line_below

    # Recurse
    outer_line_below = quick_hull(furthest_point, line_below, rec_point_list)

    return outer_line_above + outer_line_below


def show_output(tuple_list):
    """
    Format output so it's easier to read
    :param tuple_list: list of tuple indicate line
    :type tuple_list: list
    """

    final = []
    for point_1, point_2 in tuple_list:
        line = ((point_1['x'], point_1['y']), (point_2['x'], point_2['y']))
        final.append(line)

    print("Convex Hull")
    for item in final:
        print(item)


def draw(point_list, tuple_list):
    """
    Display visualization
    :param point_list: list of point at start
    :type point_list: DataFrame
    :param tuple_list: list of tuples indicating line
    :type tuple_list: List
    """

    # Setup
    fig = plt.figure(1)
    canvas = fig.add_subplot(111, facecolor='#FFFFFF')
    fig.canvas.draw()

    # Create init scatter
    x = point_list['x']
    y = point_list['y']
    canvas.scatter(x, y, color="#DC143C")

    # Parse line tuple
    final_x = []
    final_y = []
    for point_1, point_2 in tuple_list:
        # Append line edge to list
        final_x.append(point_1['x'])
        final_x.append(point_2['x'])
        final_y.append(point_1['y'])
        final_y.append(point_2['y'])

        # Create line
        list_x = [point_1['x'], point_2['x']]
        list_y = [point_1['y'], point_2['y']]
        canvas.plot(list_x, list_y, color="#ffa632")

    # Create line edge scatter
    canvas.scatter(final_x, final_y, color="#ffa632")
    # Maintains fixed aspect ratio
    canvas.set_aspect('equal')

    plt.title('Convex Hull')
    plt.show()


if __name__ == '__main__':
    # Circular figure
    # circ_fig()

    # Initialize point
    point_list = generate_points()

    # Find the left and right most
    left_most = left_most_point(point_list)
    right_most = right_most_point(point_list)

    # Separate the rest of point into two
    point_on_line_above, point_on_line_below = filter_above_below(left_most, right_most, point_list)

    # Apply quick hull algorithm
    outer_line_above = quick_hull(left_most, right_most, point_on_line_above)
    outer_line_below = quick_hull(left_most, right_most, point_on_line_below)
    outer_final = outer_line_above + outer_line_below

    # Display output
    show_output(outer_final)
    draw(point_list, outer_final)
