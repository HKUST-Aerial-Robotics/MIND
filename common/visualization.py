import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from common.geometry import get_vehicle_vertices
from av2.map.lane_segment import LaneType, LaneMarkType


def get_band_patches(polyline, width):
    band_patches = []
    for i in range(len(polyline) - 1):
        band_patches.append(LineString([polyline[i], polyline[i + 1]]).buffer(width / 2, cap_style=2))
    return band_patches


def get_circle(mean, cov):
    return Circle(xy=mean, radius=np.sqrt(cov))


def circle_to_convex_polygon(circle, resolution=100):
    t = np.linspace(0, 2 * np.pi, resolution)
    points = [
        [
            circle.center[0] + circle.radius * np.cos(t[i]),
            circle.center[1] + circle.radius * np.sin(t[i])
        ] for i in range(len(t))
    ]
    return Polygon(points).convex_hull


def reset_ax(ax):
    ax.clear()
    ax.grid(False)
    ax.set_xticks([])  # Disable x-axis ticks
    ax.set_yticks([])  # Disable y-axis ticks
    ax.set_zticks([])  # Disable z-axis ticks
    ax.xaxis.line.set_visible(False)  # Hide the x-axis
    ax.yaxis.line.set_visible(False)  # Hide the y-axis
    ax.zaxis.line.set_visible(False)  # Hide the z-axis
    ax.set_axis_off()
    ax.set_aspect('auto')


# Function to draw a cube
def draw_cube(vertices, face_clr, edge_clr, alpha=0.25):
    # Generate the list of sides' polygons of our cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]]
    ]
    return Poly3DCollection(faces, facecolors=face_clr, linewidths=1, edgecolors=edge_clr, alpha=alpha)


def draw_footprint(vertices, face_clr, edge_clr, alpha=0.25):
    # Generate the list of sides' polygons of our cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]
    return Poly3DCollection(faces, facecolors=face_clr, linewidths=2, edgecolors=edge_clr, alpha=alpha)


def draw_agent(ax, agent, z=0.1, vis_bbox=True):
    #  get agent vertices
    vertices = get_vehicle_vertices(agent.state[0], agent.state[1], z, agent.state[3],
                                    agent.bbox.l, agent.bbox.w, agent.bbox.h)

    if vis_bbox:
        ax.add_collection3d(draw_cube(vertices, face_clr=agent.clr[0], edge_clr=agent.clr[1], alpha=0.5))
    else:
        ax.add_collection3d(draw_footprint(vertices, face_clr=agent.clr[0], edge_clr=agent.clr[1], alpha=0.5))
    #  draw the heading triangle
    lon_dir = np.array([np.cos(agent.state[3]), np.sin(agent.state[3]), 0.0])
    lat_dir = np.array([-np.sin(agent.state[3]), np.cos(agent.state[3]), 0.0])
    ego_ctr = np.zeros(3)
    ego_ctr[:2] = agent.state[:2]
    ego_ctr[2] = z
    tri_pts = np.array([ego_ctr + 0.5 * agent.bbox.l * lon_dir,
                        ego_ctr + 0.15 * agent.bbox.l * lon_dir + 0.5 * agent.bbox.w * lat_dir,
                        ego_ctr + 0.15 * agent.bbox.l * lon_dir - 0.5 * agent.bbox.w * lat_dir,
                        ego_ctr + 0.5 * agent.bbox.l * lon_dir])
    ax.plot(tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2], color=agent.clr[1], linewidth=1)


def draw_dashed_polyline(ax, polyline, z, width, segment_length, color='b'):
    # cover the 2d polyline to 3d
    polyline = np.concatenate((polyline, np.full((len(polyline), 1), z)), axis=1)
    for i in range(1, len(polyline)):
        # Start and end points of the segment
        start = polyline[i - 1]
        end = polyline[i]
        segment = end - start
        segment_norm = np.linalg.norm(segment)

        # Normalize the segment to get the direction
        segment_dir = segment / segment_norm if segment_norm != 0 else np.zeros_like(segment)

        # Calculate perpendicular vector for the width
        perp_dir = np.array([-segment_dir[1], segment_dir[0], 0])  # Perpendicular in xy plane

        # Create dashed segments
        current_pos = 0
        while current_pos < segment_norm:
            dash_end = min(current_pos + segment_length, segment_norm)
            dash_start_point = start + segment_dir * current_pos
            dash_end_point = start + segment_dir * dash_end

            # Define the vertices of the polygon (rectangle) for each dash
            vertices = [
                dash_start_point - perp_dir * width / 2,
                dash_start_point + perp_dir * width / 2,
                dash_end_point + perp_dir * width / 2,
                dash_end_point - perp_dir * width / 2
            ]

            # Create a polygon and add it to the axes
            poly = Poly3DCollection([vertices], color=color, alpha=1.0)
            ax.add_collection3d(poly)

            current_pos += 2 * segment_length  # Move to the next dash


def draw_polyline(ax, polyline, z, width, color='b'):
    # cover the 2d polyline to 3d
    polyline = np.concatenate((polyline, np.full((len(polyline), 1), z)), axis=1)
    for i in range(1, len(polyline)):
        # Start and end points of the segment
        start = polyline[i - 1]
        end = polyline[i]
        segment = end - start
        segment_norm = np.linalg.norm(segment)

        # Normalize the segment to get the direction
        segment_dir = segment / segment_norm if segment_norm != 0 else np.zeros_like(segment)

        # Calculate perpendicular vector for the width
        perp_dir = np.array([-segment_dir[1], segment_dir[0], 0])  # Perpendicular in xy plane

        # Define the vertices of the polygon (rectangle) for each dash
        vertices = [
            start - perp_dir * width / 2,
            start + perp_dir * width / 2,
            end + perp_dir * width / 2,
            end - perp_dir * width / 2
        ]

        # Create a polygon and add it to the axes
        poly = Poly3DCollection([vertices], color=color, alpha=1.0)
        ax.add_collection3d(poly)


def draw_map(ax, static_map):
    # ~ drivable area
    drivable_areas = [Polygon(da.xyz[:, 0:2]) for da in static_map.vector_drivable_areas.values()]
    drivable_area_union = unary_union(drivable_areas)
    draw_polyline(ax, np.array(drivable_area_union.exterior.xy).T, z=-0.3, width=0.2, color='silver')

    # ~ lane segments
    for lane_id, lane_segment in static_map.vector_lane_segments.items():
        # centerline
        centerline = static_map.get_lane_segment_centerline(lane_id)[:, 0:2]  # use xy
        draw_dashed_polyline(ax, centerline, z=-0.1, width=0.1, segment_length=0.5, color='lightgrey')
        # lane boundary
        for boundary, mark_type in [(lane_segment.left_lane_boundary.xyz, lane_segment.left_mark_type),
                                    (lane_segment.right_lane_boundary.xyz, lane_segment.right_mark_type)]:

            clr = None
            width = 1.0
            if mark_type in [LaneMarkType.DASH_SOLID_WHITE,
                             LaneMarkType.DASHED_WHITE,
                             LaneMarkType.DOUBLE_DASH_WHITE,
                             LaneMarkType.DOUBLE_SOLID_WHITE,
                             LaneMarkType.SOLID_WHITE,
                             LaneMarkType.SOLID_DASH_WHITE]:
                clr = 'white'
                width = width
            elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                               LaneMarkType.DASHED_YELLOW,
                               LaneMarkType.DOUBLE_DASH_YELLOW,
                               LaneMarkType.DOUBLE_SOLID_YELLOW,
                               LaneMarkType.SOLID_YELLOW,
                               LaneMarkType.SOLID_DASH_YELLOW]:
                clr = 'gold'
                width = width * 1.1

            style = 'solid'
            if mark_type in [LaneMarkType.DASHED_WHITE,
                             LaneMarkType.DASHED_YELLOW,
                             LaneMarkType.DOUBLE_DASH_YELLOW,
                             LaneMarkType.DOUBLE_DASH_WHITE]:
                style = (0, (5, 10))  # loosely dashed
            elif mark_type in [LaneMarkType.DASH_SOLID_YELLOW,
                               LaneMarkType.DASH_SOLID_WHITE,
                               LaneMarkType.DOUBLE_SOLID_YELLOW,
                               LaneMarkType.DOUBLE_SOLID_WHITE,
                               LaneMarkType.SOLID_YELLOW,
                               LaneMarkType.SOLID_WHITE,
                               LaneMarkType.SOLID_DASH_WHITE,
                               LaneMarkType.SOLID_DASH_YELLOW]:
                style = 'solid'

            if (clr is not None) and (style is not None):
                ax.plot(boundary[:, 0],
                        boundary[:, 1],
                        0.0 * boundary[:, 1] - 0.1,
                        color=clr,
                        alpha=1.0,
                        linewidth=width,
                        linestyle=style)


def draw_scen_trees(ax, scen_trees):  # convert vis data to local frame
    tree_trajs = []
    # for idx, tree in enumerate(self.frames[tree_frame_idx]['scen_tree']):
    for idx, tree in enumerate(scen_trees):
        tree_traj = []
        for node in tree.get_leaf_nodes():
            full_trajs = {}
            while True:
                prob, trajs, covs, tgt_pts = node.data
                # for max sigma
                for agt_idx, (traj, cov) in enumerate(zip(trajs[1:], covs[1:])):
                    if agt_idx not in full_trajs:
                        full_trajs[agt_idx] = [traj, cov]
                    else:
                        full_trajs[agt_idx][0] = np.concatenate((traj, full_trajs[agt_idx][0]), axis=0)
                        full_trajs[agt_idx][1] = np.concatenate((cov, full_trajs[agt_idx][1]), axis=0)
                if node.parent_key is None:
                    break
                node = tree.get_node(node.parent_key)

            for agt_idx, (traj, cov) in full_trajs.items():
                polygons = []
                for ii, (pos, r) in enumerate(zip(traj, cov)):
                    if ii % 2 == 0:
                        continue
                    polygons.append(circle_to_convex_polygon(get_circle(pos, r)))

                # first compute the convex polygon of the nearby polygons
                convex_polygons = []
                for iii in range(len(polygons) - 1):
                    convex_polygons.append(polygons[iii].union(polygons[iii + 1]).convex_hull)

                tree_traj.append(unary_union(convex_polygons).exterior.xy)
        tree_trajs.append(tree_traj)
    for traj in tree_trajs[0]:
        traj = np.array(traj)
        # convert polygon exterior matplotlib patch
        verts = [list(zip(traj[0, :], traj[1, :], np.full_like(traj[0, :], 0.1)))]
        # Create a polygon and add it to the axes
        poly = Poly3DCollection(verts, alpha=0.25, facecolor='skyblue')
        ax.add_collection3d(poly)


def draw_traj_trees(ax, trees):
    z_offset = 0.1
    colors = ['skyblue']
    for idx, tree in enumerate(trees):
        clr = colors[idx]
        # cal alpha according to the tree leaves
        total_alpha = 0.8
        alpha = total_alpha / len(tree.leaves)
        for leaf_key in tree.leaves:
            nodes = tree.retrieve_nodes_to_root(leaf_key)
            traj = []
            for node in nodes:
                traj.append(node.data[0][:2])
            traj = np.array(traj)
            # convert to band patches
            band_patches = get_band_patches(traj, 1.25)
            num_patches = len(band_patches)
            for i in range(num_patches):
                verts = band_patches[i].exterior.coords.xy
                verts_3d = [list(zip(verts[0], verts[1], np.full_like(verts[0], z_offset)))]
                ax.add_collection3d(Poly3DCollection(verts_3d, alpha=alpha, facecolors=clr, edgecolors=None))


def draw_traj(ax, traj, width=0.5, clr='mediumpurple'):
    z_offset = 0.1
    band_patches = get_band_patches(traj, width)
    num_patches = len(band_patches)
    for i in range(num_patches):
        alpha = i / num_patches
        verts = band_patches[i].exterior.coords.xy
        if len(verts[0]) < 2:
            continue
        verts_3d = [list(zip(verts[0], verts[1], np.full_like(verts[0], z_offset)))]
        ax.add_collection3d(Poly3DCollection(verts_3d, alpha=alpha, facecolors=clr, edgecolors=None))
