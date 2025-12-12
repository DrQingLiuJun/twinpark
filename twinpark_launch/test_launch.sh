#!/bin/bash
# Simple test script to verify launch files

echo "Testing TwinPark Launch Files..."
echo "================================"

# Check if launch files exist
echo ""
echo "Checking launch files..."
for file in demo_twinpark.launch demo_py_impl.launch demo_cpp_impl.launch; do
    if [ -f "launch/$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done

# Check if config file exists
echo ""
echo "Checking config files..."
if [ -f "config/demo.yaml" ]; then
    echo "✓ demo.yaml exists"
else
    echo "✗ demo.yaml missing"
fi

# Check if README exists
echo ""
echo "Checking documentation..."
if [ -f "README.md" ]; then
    echo "✓ README.md exists"
else
    echo "✗ README.md missing"
fi

echo ""
echo "================================"
echo "Basic file structure check complete!"
echo ""
echo "To test the launch files with ROS:"
echo "  source ../../devel/setup.bash"
echo "  roslaunch twinpark_launch demo_py_impl.launch"
