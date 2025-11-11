#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

// Include the custom service header
// #include "ur_draw_cmake/srv/move_to_pose.hpp"
#include "ur_draw_cmake/srv/draw_stroke.hpp"

#include <vector> // Ensure this is included
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>


// Use an alias for the service type for cleaner code
using DrawStroke = ur_draw_cmake::srv::DrawStroke;
using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

// Define the name of the planning group for the UR5
const std::string PLANNING_GROUP = "ur_manipulator";
const float RAISED_PEN_HEIGHT = 0.01;
const double DISTANCE_THRESHOLD = 0.1;

// Function to calculate Euclidean distance between two poses
double calculate_distance(const geometry_msgs::msg::Pose& pose1,
                         const geometry_msgs::msg::Pose& pose2)
{
    double dx = pose2.position.x - pose1.position.x;
    double dy = pose2.position.y - pose1.position.y;
    double dz = pose2.position.z - pose1.position.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Function to get current pose from move_group in world frame
geometry_msgs::msg::Pose get_current_pose(MoveGroupInterface& move_group_interface,
                                          const rclcpp::Logger& logger)
{
    geometry_msgs::msg::PoseStamped current_pose_stamped =
        move_group_interface.getCurrentPose("tool0");

    return current_pose_stamped.pose;
}

// Assuming the service name is still DrawStroke, but now accepts a vector
void handle_stroke_service(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<rmw_request_id_t>& header,
    const DrawStroke::Request::SharedPtr request,
    DrawStroke::Response::SharedPtr response)
{
    (void)header;
    auto const logger = rclcpp::get_logger("moveit_pose_service_server");

    // Check for empty list
    if (request->target_poses.empty()) {
        RCLCPP_WARN(logger, "Received an empty list of target poses.");
        response->success = true;
        response->message = "Received empty pose list. No movement executed.";
        return;
    }

    auto move_group_interface = MoveGroupInterface(node, PLANNING_GROUP);
    move_group_interface.setPoseReferenceFrame("world");
    move_group_interface.setNumPlanningAttempts(20);

    geometry_msgs::msg::Pose current_pose = get_current_pose(move_group_interface, logger);

    const auto& first_pose = request->target_poses[0];
    auto raised_pose = first_pose;
    raised_pose.position.z += RAISED_PEN_HEIGHT;

    // Calculate distance to first target pose
    double distance_to_target = calculate_distance(current_pose, raised_pose);

    RCLCPP_INFO(logger, "Distance to first target: %.4f m (threshold: %.4f m)",
                distance_to_target, DISTANCE_THRESHOLD);

    // PHASE 1: Move to the FIRST Pose
    bool used_standard_planning = false;
    if (distance_to_target > DISTANCE_THRESHOLD) {
        RCLCPP_INFO(logger, "Phase 1: Distance exceeds threshold. Using standard planning to approach start pose...");
        move_group_interface.setPoseTarget(raised_pose, "tool0");

        moveit::planning_interface::MoveGroupInterface::Plan initial_plan;
        bool success = static_cast<bool>(move_group_interface.plan(initial_plan));

        if (!success || move_group_interface.execute(initial_plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            response->success = false;
            response->message = "Failed to reach the raised initial pose with standard planning.";
            RCLCPP_ERROR(logger, "%s", response->message.c_str());
            return;
        }

        used_standard_planning = true;
        RCLCPP_INFO(logger, "Successfully reached raised start position.");
    } else {
        RCLCPP_INFO(logger, "Phase 1: Distance within threshold. Skipping standard planning, will use Cartesian path.");
    }

    // -----------------------------------------------------------
    // PHASE 2: Cartesian Path for remaining Poses (Drawing)
    // -----------------------------------------------------------
    if (request->target_poses.size() > 1) {
        RCLCPP_INFO(logger, "Phase 2: Computing Cartesian path for %zu waypoints...", request->target_poses.size() - 1);

        // 1. Setup Orientation Constraint
        moveit_msgs::msg::OrientationConstraint ocm;
        ocm.link_name = "tool0";
        ocm.header.frame_id = move_group_interface.getPlanningFrame(); // "world"

        // Set tolerances:
        ocm.absolute_x_axis_tolerance = 0.01;
        ocm.absolute_y_axis_tolerance = 0.01;
        ocm.absolute_z_axis_tolerance = M_PI;
        ocm.weight = 1.0;

        // The constraint should be based on the *desired* orientation for the path.
        // We will use the orientation of the *last successful pose* (first_pose)
        ocm.orientation = first_pose.orientation;

        moveit_msgs::msg::Constraints path_constraints;
        path_constraints.orientation_constraints.push_back(ocm);
        move_group_interface.setPathConstraints(path_constraints);

        // 2. Prepare Waypoints
        std::vector<geometry_msgs::msg::Pose> waypoints;
        // The Cartesian planner starts at the robot's *current* pose, which is first_pose.
        // We start the waypoint list from the SECOND requested pose onwards.
        // The first waypoint must be the robot's current pose, but since we are continuing
        // the motion, we can just pass the subsequent target poses as waypoints.
        // The current pose is implicitly the start of the Cartesian path search.

        if (used_standard_planning) {
            waypoints.push_back(raised_pose); // Lower to actual first pose
        }

        // We add all poses from index 0 to the end of the list.
        for (size_t i = 0; i < request->target_poses.size(); ++i) {
            waypoints.push_back(request->target_poses[i]);
        }
        auto final_waypoint_raised = request->target_poses[request->target_poses.size() - 1];
        final_waypoint_raised.position.z+=RAISED_PEN_HEIGHT;
        waypoints.push_back(final_waypoint_raised);

        // 3. Compute the path
        moveit_msgs::msg::RobotTrajectory trajectory;
        const double eef_step = 0.005; // 5mm step
        const double jump_threshold = 0.0; // Prevent joint jumps/flips

        // Note: Cartesian Path planning does *not* automatically use the path constraints
        // set by setPathConstraints, but they are generally advisory for path validation.
        // The real continuity is enforced by jump_threshold=0.0.
        double fraction = move_group_interface.computeCartesianPath(
            waypoints,
            eef_step,
            jump_threshold,
            trajectory,
            true // avoid_collisions
        );

        move_group_interface.clearPathConstraints(); // Clear constraints immediately

        // 4. Execution of Cartesian Path
        if (fraction >= 0.9) {
            moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
            cartesian_plan.trajectory_ = trajectory; // Assign computed trajectory

            moveit::core::MoveItErrorCode execute_result = move_group_interface.execute(cartesian_plan);

            if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
                response->success = true;
                response->message = "Path execution complete and successful.";
            } else {
                response->success = false;
                response->message = "Cartesian path execution failed.";
                RCLCPP_ERROR(logger, "%s", response->message.c_str());
            }
        } else {
            response->success = false;
            response->message = "Failed to compute at least 90% of the Cartesian path.";
            RCLCPP_ERROR(logger, "Failed fraction: %.2f%%. %s", fraction * 100.0, response->message.c_str());
        }
    } else {
        // Only one pose was provided, already handled by Phase 1
        response->success = true;
        response->message = "Successfully moved to the single pose provided.";
    }
}

// Function to add a fixed collision object (The Table/Ground)
// void add_collision_object(const std::shared_ptr<rclcpp::Node>& node)
void add_collision_object()
{
    auto const logger = rclcpp::get_logger("moveit_pose_service_server");
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // Create the collision object message
    moveit_msgs::msg::CollisionObject table;
    table.header.frame_id = "world"; // The object is defined relative to the world frame
    table.id = "table";

    // Define the dimensions of the table as a box (e.g., a large, thin plane)
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;

    // Example Dimensions: 2m x 2m x 0.05m (Height)
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 2.0; // X length
    primitive.dimensions[1] = 2.0; // Y width
    primitive.dimensions[2] = 0.05; // Z height (Keep thin)

    // Define the pose (location) of the table
    geometry_msgs::msg::Pose table_pose;
    // Assuming the ground plane is at Z=0.0 in Gazebo.
    // Set the box center at Z = -height/2 to make the top surface at Z=0.0
    table_pose.orientation.w = 1.0;
    table_pose.position.x = 0.0;
    table_pose.position.y = 0.0;
    table_pose.position.z = -0.05; // Center of 0.05m box should be at Z=-0.025

    table.primitives.push_back(primitive);
    table.primitive_poses.push_back(table_pose);
    table.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Add the object to the scene
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.push_back(table);
    planning_scene_interface.addCollisionObjects(collision_objects);

    RCLCPP_INFO(logger, "Added 'table' collision object to planning scene at Z=0.0.");
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "moveit_draw_stroke_server",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    add_collision_object();

    // Create the Service Server using the custom DrawStroke service type
    rclcpp::Service<DrawStroke>::SharedPtr service =
        node->create_service<DrawStroke>(
            "moveit_draw_stroke", // The name of your service
            [&node](
                const std::shared_ptr<rmw_request_id_t>& header,
                const DrawStroke::Request::SharedPtr request,
                const DrawStroke::Response::SharedPtr response)
            {
                // Pass all four arguments to the updated handler
                handle_stroke_service(node, header, request, response);
            });
    RCLCPP_INFO(rclcpp::get_logger("moveit_pose_service_server"), "DrawStroke Service Server is ready at /moveit_draw_stroke.");

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}