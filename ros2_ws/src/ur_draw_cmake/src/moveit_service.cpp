#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

// Include the custom service header
#include "ur_draw_cmake/srv/move_to_pose.hpp"

// Use an alias for the service type for cleaner code
using MoveToPose = ur_draw_cmake::srv::MoveToPose;
using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

// Define the name of the planning group for the UR5
const std::string PLANNING_GROUP = "ur_manipulator";

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

    // Set the target Pose from the service request
    move_group_interface.setPoseTarget(target_pose);

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