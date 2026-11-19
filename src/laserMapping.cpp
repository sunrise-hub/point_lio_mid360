#include <omp.h>
#include <mutex>
#include <cmath>
#include <thread>
#include <fstream>
#include <csignal>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

#include "parameters.h"
#include "Estimator.h"


#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;

// 全局共享数据：
// ROS 回调负责把 LiDAR/IMU 写入缓冲区，主循环负责消费；
// mtx_buffer 用来保护下面这些跨线程访问的容器。
mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;

int feats_down_size = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;

int frame_ct = 0;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;

shared_ptr<ImuProcess> p_imu(new ImuProcess());
bool init_map = false, flg_first_scan = true;
PointCloudXYZI::Ptr ptr_con(new PointCloudXYZI());

// Time Log Variables
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool lidar_pushed = false, flg_reset = false, flg_exit = false;

vector<BoxPointType> cub_needrm;

deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_deque;

//surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;

MeasureGroup Measures;

sensor_msgs::msg::Imu imu_last, imu_next;
sensor_msgs::msg::Imu::ConstSharedPtr imu_last_ptr;
nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

auto logger = rclcpp::get_logger("laserMapping");

// 信号处理：收到 Ctrl-C 等退出信号时，设置退出标志并唤醒等待线程。
void SigHandle(int sig) {
    // 通知主循环退出，并唤醒可能在等待数据的线程
    flg_exit = true;
    RCLCPP_WARN(logger, "catch sig %d", sig);
    sig_buffer.notify_all();
}

// 调试用：把当前滤波器状态（姿态、位置、速度、零偏、重力）按固定列格式写入日志。
inline void dump_lio_state_to_log(FILE *fp) {
    V3D rot_ang;
    // 根据当前滤波器模式选择状态来源
    if (!use_imu_as_input) {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    } else {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }

    // 列顺序：相对时间、欧拉角、位置、角速度、速度、加速度、零偏、重力
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input) {
        // IMU 作为输入时，状态里没有单独的 omg/acc，对应列写 0
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a  
    } else {
        // IMU 作为量测时，omg/acc 也在状态里，但当前日志列仍按原格式留 0
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1),
                kf_output.x_.gravity(2)); // Bias_a
    }
    // 每条状态一行，并立即刷新，方便运行时 tail 查看
    fprintf(fp, "\r\n");
    fflush(fp);
}

// 把 LiDAR 系下的点转换到 IMU/body 系：
// 外参在线估计时使用状态里的 offset_R_L_I/offset_T_L_I，
// 否则使用配置中的 Lidar_R_wrt_IMU/Lidar_T_wrt_IMU。
void pointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    // 取出 LiDAR 系下的三维点
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    // 外参在线估计时，使用滤波器状态里的 LiDAR->IMU 外参
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_body_imu = kf_output.x_.offset_R_L_I.normalized() * p_body_lidar + kf_output.x_.offset_T_L_I;
        } else {
            p_body_imu = kf_input.x_.offset_R_L_I.normalized() * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    } else {
        // 外参固定时，直接使用配置文件读入的常值外参
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    // 写回 PCL 点，intensity 原样保留
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

int points_cache_size = 0;

// 调试/统计用：取回 ikd-Tree 已经删除的点数量，不影响主定位流程。
void points_cache_collect() // seems for debug
{
    PointVector points_history;
    // 把 ikd-Tree 内部记录的被删除点取出，只统计数量
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

// 维护局部地图立方体：
// 当 LiDAR 接近当前立方体边界时，整体平移立方体，
// 并删除移出立方体的旧地图点，控制内存和最近邻搜索范围。
void lasermap_fov_segment() {
    // 清空上一轮待删除的地图盒子
    cub_needrm.shrink_to_fit();

    // 当前 LiDAR 在世界系中的位置 = IMU 位姿 + 外参平移
    V3D pos_LiD;
    if (use_imu_as_input) {
        pos_LiD = kf_input.x_.pos + kf_input.x_.rot.normalized() * Lidar_T_wrt_IMU;
    } else {
        pos_LiD = kf_output.x_.pos + kf_output.x_.rot.normalized() * Lidar_T_wrt_IMU;
    }
    // 第一次调用：以当前位置为中心建立局部地图立方体
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    // 计算当前位置到立方体六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    // 没靠近任何边界时，地图不移动
    if (!need_move) return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    // 立方体一次移动的距离，保证移动后当前位置离边界足够远
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    // 靠近哪一侧，就把立方体向反方向平移，并记录被移出的旧盒子
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    // 统计删除点，并真正从 ikd-Tree 中删除移出局部地图的点
    points_cache_collect();
    if (cub_needrm.size() > 0) int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}

// 通用 PointCloud2 回调：
// 预处理后按 cut_frame/con_frame 配置组织成帧，写入 LiDAR 缓冲区。
void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 回调运行在 ROS executor 线程，需要和主循环互斥访问缓冲区
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();

    // 时间戳回退通常意味着驱动重启或 rosbag 循环播放，直接丢弃该帧
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
        RCLCPP_ERROR(logger, "lidar loop back, clear buffer");
        // lidar_buffer.shrink_to_fit();

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = msg->header.stamp.sec;

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());
    double time_div = get_time_sec(msg->header.stamp);

    // 把原始 PointCloud2 转成算法内部点云，并写入逐点时间
    p_pre->process(msg, ptr);
    if (cut_frame) {
        // 按逐点时间排序，便于按时间间隔切成子帧
        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++) {
            ptr_div->push_back(ptr->points[i]);
            // cout << "check time:" << ptr->points[i].curvature << endl;
            // 当前子帧已经覆盖超过 cut_frame_time_interval，就提交一个子帧
            if (ptr->points[i].curvature / double(1000) + get_time_sec(msg->header.stamp) - time_div >
                cut_frame_time_interval) {
                if (ptr_div->size() < 1) continue;
                PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
                *ptr_div_i = *ptr_div;
                lidar_buffer.push_back(ptr_div_i);
                time_buffer.push_back(time_div);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }
        // 把最后不足一个时间间隔的残余点也提交
        if (!ptr_div->empty()) {
            lidar_buffer.push_back(ptr_div);
            // ptr_div->clear();
            time_buffer.push_back(time_div);
        }
    } else if (con_frame) {
        // 拼帧模式：累计 con_frame_num 帧后合并成一帧
        if (frame_ct == 0) {
            time_con = last_timestamp_lidar; //get_time_sec(msg->header.stamp);
        }
        if (frame_ct < con_frame_num) {
            for (int i = 0; i < ptr->size(); i++) {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct++;
        } else {
            PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    } else {
        // 默认模式：一帧 LiDAR 直接入缓冲
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(get_time_sec(msg->header.stamp));
    }

    // 记录预处理耗时，释放锁并唤醒主循环
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// Livox CustomMsg 回调（Mid-360 走这条路径）：
// CustomMsg 带逐点 offset_time，Preprocess 会把它写进 curvature（ms），
// 然后按 cut_frame/con_frame 配置写入 LiDAR 缓冲区。
 void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
     // 与主循环互斥访问 LiDAR 缓冲
     mtx_buffer.lock();
     double preprocess_start_time = omp_get_wtime();
     scan_count++;

     // 时间戳回退检查
     if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
         RCLCPP_ERROR(logger, "lidar loop back, clear buffer");

         mtx_buffer.unlock();
         sig_buffer.notify_all();
         return;
     }

     last_timestamp_lidar = get_time_sec(msg->header.stamp);

     PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
     PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());

     // Livox CustomMsg -> 内部点云；curvature 保存逐点相对时间
     p_pre->process(msg, ptr);
     double time_div = get_time_sec(msg->header.stamp);
     if (cut_frame) {
         // 切帧模式：按时间间隔把一帧拆成多个子帧
         sort(ptr->points.begin(), ptr->points.end(), time_list);

         for (int i = 0; i < ptr->size(); i++) {
             ptr_div->push_back(ptr->points[i]);
             // 当前子帧达到时间间隔，提交一个子帧
             if (ptr->points[i].curvature / double(1000) + get_time_sec(msg->header.stamp) - time_div >
                 cut_frame_time_interval) {
                 if (ptr_div->size() < 1) continue;
                 PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
                 // cout << "ptr div num:" << ptr_div->size() << endl;
                 *ptr_div_i = *ptr_div;
                 // cout << "ptr div i num:" << ptr_div_i->size() << endl;
                 lidar_buffer.push_back(ptr_div_i);
                 time_buffer.push_back(time_div);
                 time_div += ptr->points[i].curvature / double(1000);
                 ptr_div->clear();
             }
         }
         // 提交最后一个残余子帧
         if (!ptr_div->empty()) {
             lidar_buffer.push_back(ptr_div);
             // ptr_div->clear();
             time_buffer.push_back(time_div);
         }
     } else if (con_frame) {
         // 拼帧模式：累计指定帧数后合成一帧
         if (frame_ct == 0) {
             time_con = last_timestamp_lidar; //get_time_sec(msg->header.stamp);
         }
         if (frame_ct < con_frame_num) {
             for (int i = 0; i < ptr->size(); i++) {
                 ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                 ptr_con->push_back(ptr->points[i]);
             }
             frame_ct++;
         } else {
             PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
             *ptr_con_i = *ptr_con;
             double time_con_i = time_con;
             lidar_buffer.push_back(ptr_con_i);
             time_buffer.push_back(time_con_i);
             ptr_con->clear();
             frame_ct = 0;
         }
     } else {
         // 默认模式：整帧入缓冲
         lidar_buffer.emplace_back(ptr);
         time_buffer.emplace_back(get_time_sec(msg->header.stamp));
     }

     // 记录耗时，释放锁并唤醒主循环
     s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
     mtx_buffer.unlock();
     sig_buffer.notify_all();
}

// IMU 回调：
// 先按 time_lag_imu_to_lidar 修正时间戳，检查时间是否回退，
// 然后把 IMU 样本压入 imu_deque，等待主循环按 LiDAR 点时间消费。
void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg_in) {
    publish_count++;

    // 复制一份消息，后面只修改副本的时间戳
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    // 把 IMU 时间换算到 LiDAR 时间轴
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_lag_imu_to_lidar);
    double timestamp = get_time_sec(msg->header.stamp);

    // 与主循环互斥访问 IMU 队列
    mtx_buffer.lock();

    // 时间回退说明数据流异常，丢弃该样本
    if (timestamp < last_timestamp_imu) {
        RCLCPP_ERROR(logger, "imu loop back, clear deque");
        // imu_deque.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    // 正常样本入队，并更新最新 IMU 时间
    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 同步一帧 LiDAR 和该帧结束前需要的 IMU，组装成 MeasureGroup：
// - LiDAR 帧为空时直接丢弃并返回 false；
// - 必须等 IMU 时间覆盖到 lidar_end_time；
// - IMU 初始化阶段会把 IMU 样本放进 meas.imu。
bool sync_packages(MeasureGroup &meas) {
    // 没有 IMU 的模式：只按 LiDAR 帧同步
    if (!imu_en) {
        if (!lidar_buffer.empty()) {
            // 取队首 LiDAR 帧及其帧头时间
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();

            // 空帧直接丢弃
            if (meas.lidar->points.size() < 1) {
                cout << "lose lidar" << std::endl;
                return false;
            }

            // 找该帧最后一个点的相对时间，换算成帧尾绝对时间
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt: meas.lidar->points) {
                if (pt.curvature > end_time) {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            meas.lidar_last_time = lidar_end_time;
            return true;
        }
        return false;
    }

    // 有 IMU 的模式：LiDAR 和 IMU 缓冲区都必须有数据
    if (lidar_buffer.empty() || imu_deque.empty()) {
        return false;
    }

    // 第一次进入时，把队首 LiDAR 帧挂到 meas 上，并计算帧尾时间
    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();
        if (meas.lidar->points.size() < 1) {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        double end_time = meas.lidar->points.back().curvature;
        for (auto pt: meas.lidar->points) {
            if (pt.curvature > end_time) {
                end_time = pt.curvature;
            }
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;
    }

    // IMU 还没覆盖到该帧帧尾，先等待更多 IMU 数据
    if (last_timestamp_imu < lidar_end_time) {
        return false;
    }

    // IMU 初始化阶段：把该帧结束前的 IMU 样本放进 meas.imu
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_) {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    } else if (!init_map) {
        // 初始地图阶段：ImuProcess 仍需要 IMU 样本做初始化/预处理
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    // 正常阶段：IMU 由主循环逐点预测时消费，这里只弹出已处理的 LiDAR 帧
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;

// 把当前帧的世界系点增量写入 ikd-Tree：
// 根据 map 体素大小判断是否需要在当前体素内新增点，
// 最终分成“需要下采样检查”和“直接插入”两组加入地图。
void map_incremental() {
    // PointToAdd：需要在当前地图体素内做判重的点
    // PointNoNeedDownsample：附近没有点、可以直接插入的点
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++) {
        if (!Nearest_Points[i].empty()) {
            // 当前点附近已经有地图点，需要做体素判重
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point;
            // 计算当前点所在的地图体素中心
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            /* If the nearest points is definitely outside the downsample box */
            // 最近点离当前体素很远，说明这个体素是空的，直接加入
            if (fabs(points_near[0].x - mid_point.x) > 1.732 * filter_size_map_min ||
                fabs(points_near[0].y - mid_point.y) > 1.732 * filter_size_map_min ||
                fabs(points_near[0].z - mid_point.z) > 1.732 * filter_size_map_min) {
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
            /* Check if there is a point already in the downsample box */
            // 检查当前体素内是否已有点，有则不重复加入
            float dist = calc_dist<float>(feats_down_world->points[i], mid_point);
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                /* Those points which are outside the downsample box should not be considered. */
                if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min &&
                    fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min &&
                    fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.emplace_back(feats_down_world->points[i]);
        } else {
            // 附近没有最近邻，直接插入，不需要做体素判重
            // PointToAdd.emplace_back(feats_down_world->points[i]);
            PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
        }
    }
    // 分别插入 ikd-Tree；第二组不再做内部下采样检查
    int add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

// 初始地图建成后调用一次：
// 把整棵 ikd-Tree 展开成点云，并发布到 /Laser_map。
void publish_init_kdtree(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes) {
    
    // odom_only 模式不需要发布地图
    if (odom_only) {return;}

    int size_init_ikdtree = ikdtree.size();
    PointCloudXYZI::Ptr laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    // 把整棵 ikd-Tree 展开到连续点云存储
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);

    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);

    // 初始地图使用当前帧尾时间戳和世界系 frame
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_header_frame_id;
    if (!odom_only) {
        pubLaserCloudFullRes->publish(laserCloudmsg);
    }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

// 发布当前帧的世界系点云（/cloud_registered）；
// 同时如果开启 pcd_save_en，就把本帧点累积到 PCD 缓存。
void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes) {

    // odom_only 模式只输出里程计，不输出点云
    if (odom_only) {return;}

    if (scan_pub_en) {
        // 把当前帧的世界系点拷贝到发布用点云
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            // if (i % 3 == 0)
            // {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; // 
            // }
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        // 发布时间戳与 frame 后发布 /cloud_registered
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = odom_header_frame_id;
        pubLaserCloudFullRes->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) {
        // 把当前帧世界系点累加到 PCD 缓存
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        // 达到保存间隔后写一个 PCD 文件并清空缓存
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

// 发布当前帧的 body/IMU 系点云（/cloud_registered_body）。
void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFull_body) {

    // odom_only 模式不发布 body 点云
    if (odom_only) {return;}

    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // 逐点把 LiDAR 系坐标转换到 IMU/body 系
    for (int i = 0; i < size; i++) {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    // body 点云使用固定 frame_id="body"
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 从当前滤波器状态中取出位置和姿态，写入 ROS Pose 结构。
template<typename T>
void set_posestamp(T &out) {
    // IMU 作为量测：使用 kf_output 的 30 维状态
    if (!use_imu_as_input) {
        // 位置
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        // 四元数顺序为 x/y/z/w
        out.orientation.x = kf_output.x_.rot.coeffs()[0];
        out.orientation.y = kf_output.x_.rot.coeffs()[1];
        out.orientation.z = kf_output.x_.rot.coeffs()[2];
        out.orientation.w = kf_output.x_.rot.coeffs()[3];
    } else {
        // IMU 作为输入：使用 kf_input 的 24 维状态
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        out.orientation.x = kf_input.x_.rot.coeffs()[0];
        out.orientation.y = kf_input.x_.rot.coeffs()[1];
        out.orientation.z = kf_input.x_.rot.coeffs()[2];
        out.orientation.w = kf_input.x_.rot.coeffs()[3];
    }
}

// 从当前滤波器状态中取出线速度和角速度，写入 ROS Twist 结构。
template<typename T>
void set_twist(T &out) {
    // 输出模式：线速度来自状态 vel，角速度来自状态 omg
    if (!use_imu_as_input) {
        /*
        第一步：读取姿态 kf_output.x_.rot 它是 SO3 旋转对象，底层继承了 Eigen 四元数类型，表示 IMU 系到世界系 的旋转
        第二步：取得归一化的姿态 .normalized()  返回归一化后的四元数值，确保它适合表示旋转。
        第三步：转换成旋转矩阵，再取转置  toRotationMatrix().transpose() toRotationMatrix()：得到一个 3×3 旋转矩阵，方向是 IMU 系 → 世界系。transpose()：取矩阵转置；对于旋转矩阵，这就是逆变换，方向变为世界系 → IMU 系。
        第四步：转换速度 * kf_output.x_.vel 将世界系速度转换成 IMU 系表达。
        */ 
        const V3D velocity_body =
            kf_output.x_.rot.normalized().toRotationMatrix().transpose() * kf_output.x_.vel;
        out.linear.x = velocity_body(0);
        out.linear.y = velocity_body(1);
        out.linear.z = velocity_body(2);
        out.angular.x = kf_output.x_.omg(0);
        out.angular.y = kf_output.x_.omg(1);
        out.angular.z = kf_output.x_.omg(2);
    } else {
        // 输入模式：线速度来自状态 vel，角速度直接取最近一帧 IMU 读数
        const V3D velocity_body =
            kf_input.x_.rot.normalized().toRotationMatrix().transpose() * kf_input.x_.vel;
        out.linear.x = velocity_body(0);
        out.linear.y = velocity_body(1);
        out.linear.z = velocity_body(2);
        out.angular.x = imu_last.angular_velocity.x;
        out.angular.y = imu_last.angular_velocity.y;
        out.angular.z = imu_last.angular_velocity.z;
    }
}

// 发布里程计和 TF：
// - 正常模式发布 /aft_mapped_to_init；
// - odom_only 模式发布 /odom_corrected；
// - 同时广播 camera_init -> aft_mapped 的 TF。
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped,
                      std::shared_ptr<tf2_ros::TransformBroadcaster> &tf_br) {

    // 可以看看/opt/ros/humble/share/nav_msgs/msg/Odometry.msg的消息定义 有header pose ；child_frame_id twist(速度估计及其协方差，在；child_frame_id系下！)

    // 设置 odom 的父系/子系：camera_init -> aft_mapped
    odomAftMapped.header.frame_id = odom_header_frame_id;
    odomAftMapped.child_frame_id = odom_child_frame_id;

    // 高频输出用当前点时间，普通输出用帧尾时间
    if (publish_odometry_without_downsample) {
        odomAftMapped.header.stamp = get_ros_time(time_current);
    } else {
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    }

    // 从滤波器状态填充位姿和速度
    set_posestamp(odomAftMapped.pose.pose);
    set_twist(odomAftMapped.twist.twist);  // 填充速度

    if (odom_only){
        // odom_only 模式额外填充协方差，供下游做融合
        Matrix3d cov = kf_output.get_P().block<3, 3>(0, 0);

        // Get the position components (first 3x3)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                odomAftMapped.pose.covariance[6 * i + j] = cov(i, j);
            }
        }

        odomAftMapped.pose.covariance[21] = 0.0;    // Covariance for roll
        odomAftMapped.pose.covariance[28] = 0.0;    // Covariance for pitch
        odomAftMapped.pose.covariance[35] = 0.05;   // Covariance for yaw

        odomAftMapped.twist.covariance[0] = 0.1;    // Covariance for linear velocity on x
        odomAftMapped.twist.covariance[7] = 0.1;    // Covariance for linear velocity on y
        odomAftMapped.twist.covariance[14] = 0.0;   // Covariance for linear velocity on z
        odomAftMapped.twist.covariance[21] = 0.0;  // Covariance for angular velocity (roll)
        odomAftMapped.twist.covariance[28] = 0.0;  // Covariance for angular velocity (pitch)
        odomAftMapped.twist.covariance[35] = 0.05;  // Covariance for angular velocity (yaw)
    }

    // 发布里程计消息
    pubOdomAftMapped->publish(odomAftMapped);

    // 再广播一份同位姿的 TF，供 RViz/tf2 使用
    //static tf2_ros::TransformBroadcaster br = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = odom_header_frame_id;
    transform.child_frame_id = odom_child_frame_id;

    transform.transform.translation.x = odomAftMapped.pose.pose.position.x;
    transform.transform.translation.y = odomAftMapped.pose.pose.position.y;
    transform.transform.translation.z = odomAftMapped.pose.pose.position.z;

    transform.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    transform.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    transform.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    transform.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;

    transform.header.stamp = odomAftMapped.header.stamp;

    tf_br->sendTransform(transform);
}

// 把当前位姿追加到轨迹 /path 并发布。
void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath) {

    // odom_only 模式不发布轨迹
    if (odom_only) {return;}

    // 更新当前位姿和时间戳
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = odom_header_frame_id;
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        // 追加到轨迹末尾并发布整条轨迹
        path.poses.emplace_back(msg_body_pose);
        pubPath->publish(path);
    }
}

// ============================================================================
// main：程序入口
//
// 总体流程：
//   1. 初始化 ROS、读取参数、初始化滤波器/地图/外参
//   2. 创建 LiDAR/IMU 订阅和各类发布器
//   3. 进入主循环：同步数据 -> IMU 初始化/重力对齐 -> 下采样/分组
//      -> 建初始地图 -> 逐点 iEKF 更新 -> 地图增量 -> 发布结果
//   4. 退出时保存 PCD（可选）并关闭日志文件
// ============================================================================
int main(int argc, char **argv) {
    // ==================== 1. ROS 初始化与参数读取 ====================
    rclcpp::init(argc, argv);
    auto nh = std::make_shared<rclcpp::Node>("laserMapping"); // 创建 ROS2 节点
    readParameters(nh);                                      // 读取 YAML/launch 参数到全局变量

    // 打印关键预处理参数，便于启动时确认配置是否生效
    cout << "lidar_type: " << lidar_type << endl;
    cout << "preprocess: scan_line=" << p_pre->N_SCANS
         << " point_filter_num=" << p_pre->point_filter_num
         << " blind=" << p_pre->blind
         << " max_iter=" << max_iter << endl;

    // ==================== 2. 轨迹与计时变量初始化 ====================
    path.header.stamp = get_ros_time(lidar_end_time);
    path.header.frame_id = odom_header_frame_id;

    /*** variables definition for counting ***/
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;
    std::time_t startTime, endTime;

    // ==================== 3. 滤波器、外参和预处理器初始化 ====================
    /*** initialize variables ***/
    double FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        } else {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }
    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    // 注册两个滤波器的预测函数、量测函数和雅可比。
    // use_imu_as_input=false 时主流程使用 kf_output；
    // use_imu_as_input=true 时主流程使用 kf_input。
    kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
    // 必须在 init_* 之后设置：init 里会把 maximum_iter 重置为 1
    kf_input.set_maximum_iter(max_iter);
    kf_output.set_maximum_iter(max_iter);
    RCLCPP_INFO(logger, "IEKF maximum_iter = %d", kf_output.get_maximum_iter());
    // 设置两个滤波器的初始协方差 P。
    // 这里只调整几个关键块，其余状态默认给一个较小的不确定度。
    Eigen::Matrix<double, 24, 24> P_init = MD(24, 24)::Identity() * 0.01;
    P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
    P_init.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
    kf_input.change_P(P_init);
    kf_output.change_P(P_init_output);
    // 过程噪声协方差 Q：输入模式和输出模式分别构造。
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
    // ==================== 4. 调试日志文件 ====================
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_out, fout_imu_pbp;
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"), ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    // ==================== 5. 创建 LiDAR/IMU 订阅 ====================
    /*** ROS subscribe initialization ***/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    if (p_pre->lidar_type == AVIA) {
        sub_pcl_livox_ = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
    } else {
    sub_pcl = nh->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
    }
    auto sub_imu = nh->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200000, imu_cbk);

    // ==================== 6. 创建发布器 ====================
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_body;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    if (!odom_only){
        pubLaserCloudFullRes = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_registered", 100000);
        pubLaserCloudFullRes_body = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_registered_body", 100000);
        pubLaserCloudEffect = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_effected", 100000);
        pubLaserCloudMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/Laser_map", 100000);
        pubPath = nh->create_publisher<nav_msgs::msg::Path>
                ("/path", 100000);
    }

    // Choose topic name depending on odom_only value
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    if (odom_only){
        pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>
                ("/odom_corrected", 100000);
    } else {
        pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>
                ("/aft_mapped_to_init", 100000);
    }

    //auto plane_pub = nh->create_publisher<visualization_msgs::msg::Marker>
    //        ("/planner_normal", 1000);
    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(nh);
//------------------------------------------------------------------------------------------------------
    // ==================== 7. 主循环 ====================
    signal(SIGINT, SigHandle);
    rclcpp::Rate rate(5000);
    while (rclcpp::ok()) {
        if (flg_exit) break;
        //ros::spinOnce();

        // 7.1 处理当前可用的 ROS 回调：
        // LiDAR/IMU 回调只负责把数据放进缓冲区，不在这里做估计。
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(nh);
        executor.spin_some(); // 处理当前可用的回调

        // 7.2 尝试凑齐“一帧 LiDAR + 该帧结束前的 IMU”。
        // 数据不完整时 sync_packages() 返回 false，主循环继续下一轮。
        if (sync_packages(Measures)) {
            // 记录第一帧 LiDAR 时间，作为后续日志和相对时间的基准。
            if (flg_first_scan) {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                cout << "first lidar time" << first_lidar_time << endl;
            }

            // 回放 rosbag 等情况下的重置标志：重置 IMU 初始化状态。
            if (flg_reset) {
                RCLCPP_WARN(logger, "reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }

            // 本帧各阶段耗时统计清零。
            double t0, t1, t2, t3, t4, t5, match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();

            // 7.3 IMU 初始化 / 点云传递。
            // 前 100 个 IMU 样本用于估计重力方向、陀螺零偏；
            // 初始化完成后，本函数把当前帧点云交给 feats_undistort。
            p_imu->Process(Measures, feats_undistort);

            if (feats_undistort->empty() || feats_undistort == nullptr) {
                continue;
            }

            // 7.4 重力对齐：只在初始化阶段做一次。
            // 静止启动时用 IMU 平均加速度估计重力；
            // 非静止启动时使用配置里的 gravity_init。
            if (imu_en) {
                if (!p_imu->gravity_align_) {
                    while (Measures.lidar_beg_time > get_time_sec(imu_next.header.stamp)) {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        // imu_deque.pop();
                    }
                    if (non_station_start) {
                        state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    } else {
                        state_in.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                        state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                        state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                    }
                    if (gravity_align) {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                        p_imu->Set_init(state_in.gravity, rot_init);
                        state_in.gravity = state_out.gravity = p_imu->gravity_;
                        state_in.rot = state_out.rot = rot_init;
                        state_in.rot.normalize();
                        state_out.rot.normalize();
                        state_out.acc = -rot_init.transpose() * state_out.gravity;
                    }
                    kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                }
            } else {
                if (!p_imu->gravity_align_) {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc *= -1;
                }
            }

            // 7.5 局部地图裁剪：雷达移动时平移地图立方体并删除远端地图点。
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            // 7.6 当前帧下采样、按逐点时间排序，并按时间戳分组。
            // 每个时间组后面的 iEKF 更新都使用该组对应的时间。
            /*** downsample the feature points in a scan ***/
            t1 = omp_get_wtime();
            if (space_down_sample) {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            } else {
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            }
            time_seq = time_compressing<int>(feats_down_body);
            feats_down_size = feats_down_body->points.size();

            // 7.7 初始地图建立阶段：
            // 把前几帧点云累计到 init_feats_world，
            // 点数达到 init_map_size 后构建 ikd-Tree。
            // 初始地图建立完成前不进入正常里程计更新。
            /*** initialize the map kdtree ***/
            if (!init_map) {
                if (ikdtree.Root_Node == nullptr) //
                    // if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                }

                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++) {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) {
                    init_feats_world->points.emplace_back(feats_down_world->points[i]);
                }
                if (init_feats_world->size() < init_map_size) continue;
                ikdtree.Build(init_feats_world->points);
                init_map = true;
                publish_init_kdtree(pubLaserCloudMap); //(pubLaserCloudFullRes);
                continue;
            }

            // 7.8 为点到面匹配准备数据结构：
            // 法向量、世界系点、最近邻点、点到 IMU 的叉乘矩阵等。
            /*** ICP and Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/ 
            // reserve() 只分配容量，不创建元素 后面直接用[i]写入可能会出现问题 应该改为resize 
            crossmat_list.resize(feats_down_size);
            pbody_list.resize(feats_down_size);
            // pbody_ext_list.reserve(feats_down_size);

            for (size_t i = 0; i < feats_down_body->size(); i++) {
                V3D point_this(feats_down_body->points[i].x,
                               feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
                pbody_list[i] = point_this;
                if (extrinsic_est_en) {
                    if (!use_imu_as_input) {
                        point_this = kf_output.x_.offset_R_L_I.normalized() * point_this + kf_output.x_.offset_T_L_I;
                    } else {
                        point_this = kf_input.x_.offset_R_L_I.normalized() * point_this + kf_input.x_.offset_T_L_I;
                    }
                } else {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list[i] = point_crossmat;
            }

            // 7.9 逐点/逐组 iEKF 更新。
            // 这里分成两种滤波器模式：
            //   !use_imu_as_input：IMU 作为量测，使用 kf_output；
            //    use_imu_as_input：IMU 作为输入，使用 kf_input。
            if (!use_imu_as_input) {
                bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++) {
                    // 当前时间组的代表点：该组的最后一个点。
                    // curvature 存的是该点相对帧头的毫秒偏移。
                    PointType &point_body = feats_down_body->points[idx + time_seq[k]];

                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    // 首次进入时，把 IMU 指针推进到当前 LiDAR 时间附近。
                    if (is_first_frame) {
                        if (imu_en) {
                            while (time_current > get_time_sec(imu_next.header.stamp)) {
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                imu_deque.pop_front();
                                // imu_deque.pop();
                            }

                            angvel_avr
                                    << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr
                                    << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }

                    // 用当前时间组之前的 IMU 样本逐步预测状态。
                    // 每处理一个 IMU 样本，同时做一次 IMU 量测更新。
                    if (imu_en) {
                        bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                        while (imu_comes) {
                            imu_upda_cov = true;
                            angvel_avr
                                    << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr
                                    << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            double dt = get_time_sec(imu_last.header.stamp) - time_predict_last_const;
                            kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = get_time_sec(imu_last.header.stamp); // big problem
                            imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                            // if (!imu_comes)
                            {
                                double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;

                                if (dt_cov > 0.0) {
                                    time_update_last = get_time_sec(imu_last.header.stamp);
                                    double propag_imu_start = omp_get_wtime();

                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                    propag_time += omp_get_wtime() - propag_imu_start;
                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                        }
                    }

                    // 从上一个预测时刻推进到当前点时间。
                    double dt = time_current - time_predict_last_const;
                    double propag_state_start = omp_get_wtime();
                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_state_start;
                    time_predict_last_const = time_current;
                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();

                    // 没有任何点时不进行雷达量测更新。
                    if (feats_down_size < 1) {
                        RCLCPP_WARN(logger, "No point, skip this scan!\n");
                        idx += time_seq[k];
                        continue;
                    }

                    // 点到面量测更新：
                    // h_model_output 内部会找 5 个近邻、拟合平面、计算残差。
                    if (!kf_output.update_iterated_dyn_share_modified()) {
                        idx = idx + time_seq[k];
                        continue;
                    }

                    if (prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (!imu_en && (dt_cov >= imu_time_inte)) // (point_cov_not_prop && imu_prop_cov)
                        {
                            double propag_cov_start = omp_get_wtime();
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            imu_upda_cov = false;
                            time_update_last = time_current;
                            propag_time += omp_get_wtime() - propag_cov_start;
                        }
                    }

                    solve_start = omp_get_wtime();

                    if (publish_odometry_without_downsample) {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log) {
                            state_out = kf_output.x_;
                            euler_cur = SO3ToEuler(state_out.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                     << euler_cur.transpose() << " " << state_out.pos.transpose() << " "
                                     << state_out.vel.transpose() << " " << state_out.omg.transpose() << " "
                                     << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " "
                                     << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                                     << feats_undistort->points.size() << endl;
                        }
                    }

                    // 更新完成后，把本组点重新变换到世界系，供地图增量使用。
                    for (int j = 0; j < time_seq[k]; j++) {
                        PointType &point_body_j = feats_down_body->points[idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[idx + j + 1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }

                    solve_time += omp_get_wtime() - solve_start;

                    update_time += omp_get_wtime() - t_update_start;
                    idx += time_seq[k];
                    // cout << "pbp output effect feat num:" << effct_feat_num << endl;
                }
            } else {
                // 另一条分支：IMU 作为状态输入，使用 kf_input（24 维状态）。
                // 这里不再把 IMU 当量测，而是直接用 IMU 读数参与状态预测。
                bool imu_prop_cov = false;
                effct_feat_num = 0;

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++) {
                    // 取当前时间组的代表点及其绝对时间。
                    PointType &point_body = feats_down_body->points[idx + time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    // 首次进入时初始化 IMU 输入和时间基准。
                    if (is_first_frame) {
                        while (time_current > get_time_sec(imu_next.header.stamp)) {
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            // imu_deque.pop();
                        }
                        imu_prop_cov = true;
                        // imu_upda_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current;
                        // if(prop_at_freq_of_imu)
                        {
                            input_in.gyro << imu_last.angular_velocity.x,
                                    imu_last.angular_velocity.y,
                                    imu_last.angular_velocity.z;

                            input_in.acc << imu_last.linear_acceleration.x,
                                    imu_last.linear_acceleration.y,
                                    imu_last.linear_acceleration.z;
                            // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                            //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                            //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);

                            // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                            //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                            // 0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);

                            // angvel_avr -= state.bias_g;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        }
                    }

                    // 用 IMU 样本把状态推进到当前点时间。
                    while (time_current > get_time_sec(imu_next.header.stamp)) // && !imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        input_in.gyro
                                << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc
                                << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                        // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                        //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                        //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);

                        // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                        //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                        //             0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        double dt = get_time_sec(imu_last.header.stamp) - t_last;

                        // if(!prop_at_freq_of_imu)
                        // {       
                        double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = get_time_sec(imu_last.header.stamp); //time_current;
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false);
                        t_last = get_time_sec(imu_last.header.stamp);
                        imu_prop_cov = true;
                        // imu_upda_cov = true;
                    }

                    // 处理完中间 IMU 后，再预测到当前点自身的时间。
                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();

                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);

                    propag_time += omp_get_wtime() - propag_start;

                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();

                    // 没有点时跳过雷达量测更新。
                    if (feats_down_size < 1) {
                        RCLCPP_WARN(logger, "No point, skip this scan!\n");

                        idx += time_seq[k];
                        continue;
                    }

                    // 点到面量测更新。
                    if (!kf_input.update_iterated_dyn_share_modified()) {
                        idx = idx + time_seq[k];
                        continue;
                    }

                    solve_start = omp_get_wtime();

                    // if(prop_at_freq_of_imu)
                    // {
                    //     double dt_cov = time_current - time_update_last;
                    //     if ((imu_prop_cov && dt_cov > 0.0) || (dt_cov >= imu_time_inte * 1.2)) 
                    //     {
                    //         double propag_cov_start = omp_get_wtime();
                    //         kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                    //         propag_time += omp_get_wtime() - propag_cov_start;
                    //         time_update_last = time_current;
                    //         imu_prop_cov = false;
                    //     }
                    // }
                    if (publish_odometry_without_downsample) {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log) {
                            state_in = kf_input.x_;
                            euler_cur = SO3ToEuler(state_in.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                     << euler_cur.transpose() << " " << state_in.pos.transpose() << " "
                                     << state_in.vel.transpose() << " " << state_in.bg.transpose() << " "
                                     << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " "
                                     << feats_undistort->points.size() << endl;
                        }
                    }

                    // 更新后把本组点重新投影到世界系。
                    for (int j = 0; j < time_seq[k]; j++) {
                        PointType &point_body_j = feats_down_body->points[idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[idx + j + 1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }
                    solve_time += omp_get_wtime() - solve_start;

                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }
            }

            // 7.10 发布本帧结果：
            // - 发布 odom/TF（按配置决定是否每帧都发）
            // - 把当前帧点加入地图
            // - 发布轨迹和点云
            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample) {
                publish_odometry(pubOdomAftMapped, tf_broadcaster);
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();

            if (feats_down_size > 4) {
                map_incremental();
            }

            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en) publish_path(pubPath);
            if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);

            // 7.11 运行日志与耗时统计（runtime_pos_log_enable=true 时启用）。
            /*** Debug variables Logging ***/
            if (runtime_pos_log) {
                frame_num++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                { aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num; }
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",
                       t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp,
                       aver_time_propag);
                if (!publish_odometry_without_downsample) {
                    if (!use_imu_as_input) {
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                 << euler_cur.transpose() << " " << state_out.pos.transpose() << " "
                                 << state_out.vel.transpose() << " " << state_out.omg.transpose() << " "
                                 << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " "
                                 << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                                 << feats_undistort->points.size() << endl;
                    } else {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                 << euler_cur.transpose() << " " << state_in.pos.transpose() << " "
                                 << state_in.vel.transpose() << " " << state_in.bg.transpose() << " "
                                 << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " "
                                 << feats_undistort->points.size() << endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
        rate.sleep();
    }

    // ==================== 8. 退出保存 ====================
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
       2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en) {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();

    return 0;
}
