#### ROS2 HUMBLE 
က Default အားဖြင့် Gazebo Fortress ဖြစ်ပြီး packages တွေက ros-humble-ign-* ဖြစ်တယ်။

#### ROS2 Iron 
က Gazebo Fortress, Gazebo Garden ( experimental ) ဖြစ်ပါတယ်။

#### ROS2 Jazzy 
က Gazebo Harmonic ဖြစ်ပြီး သူ့ရဲ့ packages တွေက ros2-jazzy-gz-* ဖြစ်ပါတယ်။


#### PX4-Autopilot release/1.14
က Gazebo Fortress ပါတဲ့။ chatgpt က ပြောတာ။ Gemini ကပြောတာကတော့ Gazebo Graden ပါတဲ့။

#### PX4-Autopilot release/1.15
က Gazebo Fortress ပါတဲ့။ chatgpt က ပြောတာ။ Gemini ကပြောတာကတော့ သေချာမပြောထားပေမဲ့ Experimental အရ Gazebo Graden ပါတဲ့။

#### PX4-Autopilot release/1.16
က Gazebo Garden ပါတဲ့။ chatgpt က ပြောတာ။ Gemini ကပြောတာကတော့ Gazebo Harmonic ပါတဲ့။

#### how to fix ကြည့်ရတာ Harmonic နဲ့ Garden driver တို့ အလုပ်တွဲလုပ်နေပုံရတယ်။

sudo apt remove ros-humble-ros-gz-* ros-iron-ros-gz-*
sudo apt install ros-humble-ros-gzgarden-*
