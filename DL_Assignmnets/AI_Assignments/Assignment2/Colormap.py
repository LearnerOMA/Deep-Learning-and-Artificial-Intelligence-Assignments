# -*- coding: utf-8 -*-
"""ColorMap.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18Vgj5CGFcOgNcFr0hV5Q8q3LZtsRxMyq
"""

# Map Coloring Problem using Backtracking with Step-by-Step Output

# Function to check if the current color assignment is safe for the vertex
def is_safe(vertex, graph, colors, color, assigned_colors):
    for i in range(len(graph)):
        if graph[vertex][i] == 1 and assigned_colors[i] == color:
            return False
    return True

# Function to print the current state of color assignments
def print_assignment(assigned_colors, step):
    print(f"Step {step}: {assigned_colors}")

# Function to solve the map coloring problem using backtracking with step-by-step output
def map_coloring(graph, colors, assigned_colors, vertex, step):
    # If all vertices are assigned a color, return True
    if vertex == len(graph):
        return True

    # Try different colors for the current vertex
    for color in colors:
        if is_safe(vertex, graph, colors, color, assigned_colors):
            # Assign the current color to the vertex
            assigned_colors[vertex] = color
            print_assignment(assigned_colors, step)
            step += 1  # Increment step for next assignment

            # Recur to assign colors to the rest of the vertices
            if map_coloring(graph, colors, assigned_colors, vertex + 1, step):
                return True

            # If assigning color didn't lead to a solution, backtrack
            assigned_colors[vertex] = None  # Backtrack
            print(f"Backtrack from Step {step}")

    return False

# Main function to solve the problem
def solve_map_coloring(graph, colors):
    assigned_colors = [None] * len(graph)  # Initialize all vertices as unassigned
    step = 1  # Start the step counter

    if not map_coloring(graph, colors, assigned_colors, 0, step):
        print("No solution exists")
        return None

    return assigned_colors


# Adjacency matrix of the map (Graph)
graph = [
    [0, 1, 0, 1],  # Region 1 is connected to Region 2 and Region 4
    [1, 0, 1, 1],  # Region 2 is connected to Region 1, Region 3, and Region 4
    [0, 1, 0, 1],  # Region 3 is connected to Region 2 and Region 4
    [1, 1, 1, 0]   # Region 4 is connected to Region 1, Region 2, and Region 3
]

# graph = [
#     [0, 1, 0, 1, 0],  # Region 1 is connected to Region 2 and Region 4
#     [1, 0, 1, 1, 1],  # Region 2 is connected to Region 1, Region 3, Region 4, and Region 5
#     [0, 1, 0, 0, 1],  # Region 3 is connected to Region 2 and Region 5
#     [1, 1, 0, 0, 1],  # Region 4 is connected to Region 1, Region 2, and Region 5
#     [0, 1, 1, 1, 0]   # Region 5 is connected to Region 2, Region 3, and Region 4
# ]


# List of available colors
colors = ['Red', 'Green', 'Blue']

# Solve the problem and print the result
solution = solve_map_coloring(graph, colors)

if solution:
    print("\nFinal Solution:", solution)