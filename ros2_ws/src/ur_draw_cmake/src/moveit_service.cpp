#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/orientation_constraint.hpp>

// Include the custom service header
#include "ur_draw_cmake/srv/move_to_pose.hpp"

// Use an alias for the service type for cleaner code
using MoveToPose = ur_draw_cmake::srv::MoveToPose;
using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

// Define the name of the planning group for the UR5
const std::string PLANNING_GROUP = "ur_manipulator";

bool first_pose = true;

// Function to handle the service request
void handle_pose_service(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::shared_ptr<rmw_request_id_t>& header,  // New argument for request ID
    const MoveToPose::Request::SharedPtr request,
    MoveToPose::Response::SharedPtr response)
{
    (void)header; // To avoid unused parameter warning if not used
    auto const logger = rclcpp::get_logger("moveit_pose_service_server");

    // Extract the target pose from the request
    auto const& target_pose = request->target_pose;

    RCLCPP_INFO(logger, "Service request received. Target Pose (x: %.2f, y: %.2f, z: %.2f)",
                target_pose.position.x, target_pose.position.y, target_pose.position.z);

    // Create the MoveIt MoveGroup Interface
    auto move_group_interface = MoveGroupInterface(node, PLANNING_GROUP);
    move_group_interface.setPoseReferenceFrame("world");

    move_group_interface.setNumPlanningAttempts(50);

    // // Set the time allowed for planning (default is 5.0 seconds)
    move_group_interface.setPlanningTime(10.0);

    // move_group_interface.setGoalJointTolerance(0.001); // E.g., 0.001 radians

    // // Set position and orientation tolerance (default is often 1e-4)
    // // This applies to the Cartesian distance (position)
    // move_group_interface.setGoalPositionTolerance(0.005);

    // const std::string planning_frame = move_group_interface.getPlanningFrame();
    // RCLCPP_INFO(logger, "MoveGroup planning frame: %s", planning_frame.c_str());

    std::string eef_link = move_group_interface.getEndEffectorLink();
    RCLCPP_INFO(node->get_logger(), "EEF Link: %s", eef_link.c_str());

    if (first_pose){
        // Set the target Pose from the service request
        move_group_interface.setPoseTarget(target_pose,"tool0");
        first_pose = false;



        // Create a plan to that target pose
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = static_cast<bool>(move_group_interface.plan(plan));

        // Execute the plan and set the service response
        if (success) {
            RCLCPP_INFO(logger, "Planning successful. Attempting execution...");
            moveit::core::MoveItErrorCode execute_result = move_group_interface.execute(plan);

            if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
                response->success = true;
                response->message = "Plan executed successfully.";
                RCLCPP_INFO(logger, "Execution complete and successful.");
            } else {
                response->success = false;
                response->message = "Planning successful but execution failed.";
                RCLCPP_ERROR(logger, "Execution failed.");
            }
        } else {
            response->success = false;
            response->message = "Planing failed for the requested pose!";
            RCLCPP_ERROR(logger, "Planing failed!");
        }
    }
    else{
        // --- Add an Orientation Constraint ---
        moveit_msgs::msg::OrientationConstraint ocm;
        ocm.link_name = "tool0"; // Constraint applies to the tool0 link
        // ocm.header.frame_id = move_group_interface.getPlanningFrame();
        ocm.header.frame_id = "wrist_2_link";
        // Use the target pose's orientation for the constraint
        ocm.orientation = target_pose.orientation;
        ocm.absolute_x_axis_tolerance = 0.01;
        ocm.absolute_y_axis_tolerance = M_PI;
        ocm.absolute_z_axis_tolerance = 0.01;
        ocm.weight = 1.0;

        moveit_msgs::msg::Constraints path_constraints;
        path_constraints.orientation_constraints.push_back(ocm);
        move_group_interface.setPathConstraints(path_constraints);

        std::vector<geometry_msgs::msg::Pose> waypoints;
        geometry_msgs::msg::Pose current_pose = move_group_interface.getCurrentPose().pose;


        // 1. Add the starting pose as the first waypoint
        waypoints.push_back(current_pose);

        // 2. Add the target pose from the service request as the next waypoint
        waypoints.push_back(request->target_pose); // Assuming request->target_pose is the target

        // 3. Compute the Cartesian path
        moveit_msgs::msg::RobotTrajectory trajectory;
        const double eef_step = 0.005; // Step size for interpolation (e.g., 1 cm)
        const double jump_threshold = 0.0; // Avoid jumps in joint space (0.0 is often safer)
        double fraction = move_group_interface.computeCartesianPath(
            waypoints,
            eef_step,
            jump_threshold,
            trajectory,
            true // optional: true to avoid collisions
        );

        RCLCPP_INFO(node->get_logger(), "Cartesian path computed (%.2f%% achieved)", fraction * 100.0);

        if (fraction >= 0.3) // Check if the path was mostly successful
        {
            // 4. Execute the resulting trajectory
            // plan.trajectory = trajectory;
            // move_group_interface.execute(plan);
            // moveit::planning_interface::MoveGroupInterface::Plan plan;
            // bool success = static_cast<bool>(move_group_interface.plan(plan));
            // move_group_interface.plan(plan);
            // response->success = true;
            // plan.trajectory_ = trajectory; // <-- This is the key link

            // moveit::core::MoveItErrorCode execute_result = move_group_interface.execute(plan);
            moveit::core::MoveItErrorCode execute_result = move_group_interface.execute(trajectory);

            if (execute_result == moveit::core::MoveItErrorCode::SUCCESS) {
                response->success = true;
                response->message = "Cartesian path executed successfully.";
                RCLCPP_INFO(logger, "Cartesian Execution complete and successful.");
            } else {
                response->success = false;
                response->message = "Cartesian planning successful but execution failed.";
                RCLCPP_ERROR(logger, "Cartesian Execution failed.");
            }
        }
        else
        {
            RCLCPP_ERROR(node->get_logger(), "Failed to compute full Cartesian path.");
            response->success = false;
        }
        // 4. IMPORTANT: Clear constraints after planning/execution!
        move_group_interface.clearPathConstraints();
    }
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>(
        "moveit_pose_service_server",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // Create the Service Server using the custom MoveToPose service type
    rclcpp::Service<MoveToPose>::SharedPtr service =
        node->create_service<MoveToPose>(
            "moveit_to_pose", // The name of your service
            [&node](
                const std::shared_ptr<rmw_request_id_t>& header,
                const MoveToPose::Request::SharedPtr request,
                const MoveToPose::Response::SharedPtr response)
            {
                // Pass all four arguments to the updated handler
                handle_pose_service(node, header, request, response);
            });
    RCLCPP_INFO(rclcpp::get_logger("moveit_pose_service_server"), "MoveToPose Service Server is ready at /moveit_to_pose.");

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}