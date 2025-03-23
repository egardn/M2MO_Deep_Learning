import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class RelationalDataset:
    def __init__(self, num_samples=10000, img_size=128):
        """
        Create a synthetic dataset with 8 classes based on relational reasoning:
        - Classes 0-3: Line parallel to one side of the square
        - Classes 4-7: Line parallel to one diagonal of the square
        - Classes 0,2,4,6: Cross has one axis parallel to line
        - Classes 1,3,5,7: Cross has one axis at 45° angle with line
        - Classes 0,1,4,5: Rectangle's long side is parallel to line
        - Classes 2,3,6,7: Rectangle's long side is perpendicular to line

        Args:
            num_samples: Total number of samples to generate
            img_size: Size of square images
            transform: PyTorch transforms to apply
        """
        self.num_samples = num_samples
        self.img_size = img_size

        # Generate data
        self.data = []
        self.targets = []

        self._generate_data()

    def _generate_data(self):
        """Generate synthetic images according to class rules"""
        samples_per_class = self.num_samples // 8

        for class_idx in range(8):
            for _ in range(samples_per_class):
                # Create blank image
                img = np.zeros((self.img_size, self.img_size), dtype=np.float32)

                # Determine key properties based on class
                is_segment_parallel_to_side = class_idx < 4  # Classes 0-3: parallel to side; Classes 4-7: parallel to diagonal
                is_cross_parallel = (class_idx % 2 == 0)  # Classes 0,2,4,6: parallel; Classes 1,3,5,7: 45° angle
                is_rectangle_parallel = (class_idx % 4 < 2)  # Classes 0,1,4,5: rectangle parallel to line; Classes 2,3,6,7: perpendicular

                # Generate image with proper properties
                img = self._create_image(img, is_segment_parallel_to_side, is_cross_parallel, is_rectangle_parallel)

                # Add noise to make task more challenging
                img = img + np.random.normal(0, 0.05, img.shape)
                img = np.clip(img, 0, 1)

                # Convert to uint8 for PIL compatibility
                img_uint8 = (img * 255).astype(np.uint8)

                self.data.append(img_uint8)
                self.targets.append(class_idx)

    def _create_image(self, img, is_segment_parallel_to_side, is_cross_parallel, is_rectangle_parallel):
        """Create image with square, line, cross, and rectangle with specified relational properties"""
        # Define minimum distance between objects to prevent overlapping
        min_distance = self.img_size * 0.12
        padding = int(self.img_size * 0.1)

        # List to store object positions and sizes for collision detection
        objects = []

        # Generate square with random position and rotation
        while True:
            # Variable square size
            square_size = np.random.uniform(self.img_size * 0.07, self.img_size * 0.13)
            square_center = (
                np.random.randint(padding + square_size//2, self.img_size - padding - square_size//2),
                np.random.randint(padding + square_size//2, self.img_size - padding - square_size//2)
            )

            # First object always accepted
            objects.append({"type": "square", "center": square_center, "size": square_size})
            break

        # Random rotation for square
        square_angle = np.random.uniform(0, np.pi/2)

        # Draw the square
        self._draw_square(img, square_center, square_size, square_angle)

        # Determine line angle based on square orientation
        if is_segment_parallel_to_side:
            # Parallel to one side of the square (0° or 90° relative to square orientation)
            side_choice = np.random.choice([0, np.pi/2])
            line_angle = square_angle + side_choice
        else:
            # Parallel to one diagonal of the square (45° or 135° relative to square orientation)
            diagonal_choice = np.random.choice([np.pi/4, 3*np.pi/4])
            line_angle = square_angle + diagonal_choice

        # Generate a line that extends to the image boundaries
        attempts = 0
        line_placed = False

        while attempts < 50 and not line_placed:
            # Random point through which the line will pass
            line_center = (
                np.random.randint(padding, self.img_size - padding),
                np.random.randint(padding, self.img_size - padding)
            )

            # Check distance to previously placed objects
            too_close = False
            for obj in objects:
                dist = np.sqrt((line_center[0] - obj["center"][0])**2 +
                              (line_center[1] - obj["center"][1])**2)
                if dist < min_distance + obj["size"]/2:
                    too_close = True
                    break

            if not too_close:
                # Calculate the line endpoints by extending to image boundaries
                # Direction vector
                dx = np.cos(line_angle)
                dy = np.sin(line_angle)

                # Find intersections with image boundaries
                # We need to solve for t in the parametric equation:
                # (x,y) = (line_center_x, line_center_y) + t * (dx, dy)
                # For each of the four boundaries: x=0, x=img_size-1, y=0, y=img_size-1

                t_values = []

                # Left boundary (x=0)
                if abs(dx) > 1e-10:  # Avoid division by zero
                    t_left = -line_center[0] / dx
                    y_left = line_center[1] + t_left * dy
                    if 0 <= y_left < self.img_size:
                        t_values.append((t_left, (0, int(y_left))))

                # Right boundary (x=img_size-1)
                if abs(dx) > 1e-10:
                    t_right = (self.img_size - 1 - line_center[0]) / dx
                    y_right = line_center[1] + t_right * dy
                    if 0 <= y_right < self.img_size:
                        t_values.append((t_right, (self.img_size-1, int(y_right))))

                # Top boundary (y=0)
                if abs(dy) > 1e-10:
                    t_top = -line_center[1] / dy
                    x_top = line_center[0] + t_top * dx
                    if 0 <= x_top < self.img_size:
                        t_values.append((t_top, (int(x_top), 0)))

                # Bottom boundary (y=img_size-1)
                if abs(dy) > 1e-10:
                    t_bottom = (self.img_size - 1 - line_center[1]) / dy
                    x_bottom = line_center[0] + t_bottom * dx
                    if 0 <= x_bottom < self.img_size:
                        t_values.append((t_bottom, (int(x_bottom), self.img_size-1)))

                # Sort by parameter t to get the correct order
                t_values.sort()

                # We need exactly 2 intersection points
                if len(t_values) >= 2:
                    # Take the first and last intersection points
                    line_start = t_values[0][1]
                    line_end = t_values[-1][1]

                    # Record the center point for collision detection with other objects
                    line_size = np.sqrt((line_end[0] - line_start[0])**2 +
                                      (line_end[1] - line_start[1])**2)
                    objects.append({"type": "line", "center": line_center, "size": line_size})
                    line_placed = True

                    # Draw the full line
                    cv2.line(img, line_start, line_end, 1.0, 2)

            attempts += 1

        # If we couldn't place a line after max attempts, use a fallback approach
        if not line_placed:
            # Fallback: use entire diagonal of the image
            if abs(np.cos(line_angle)) > abs(np.sin(line_angle)):
                # More horizontal
                if np.cos(line_angle) > 0:
                    line_start = (0, int(self.img_size/2 - np.tan(line_angle) * self.img_size/2))
                    line_end = (self.img_size-1, int(self.img_size/2 + np.tan(line_angle) * self.img_size/2))
                else:
                    line_start = (self.img_size-1, int(self.img_size/2 + np.tan(line_angle) * self.img_size/2))
                    line_end = (0, int(self.img_size/2 - np.tan(line_angle) * self.img_size/2))
            else:
                # More vertical
                if np.sin(line_angle) > 0:
                    line_start = (int(self.img_size/2 - self.img_size/(2*np.tan(line_angle))), 0)
                    line_end = (int(self.img_size/2 + self.img_size/(2*np.tan(line_angle))), self.img_size-1)
                else:
                    line_start = (int(self.img_size/2 + self.img_size/(2*np.tan(line_angle))), self.img_size-1)
                    line_end = (int(self.img_size/2 - self.img_size/(2*np.tan(line_angle))), 0)

            # Draw the line
            cv2.line(img, line_start, line_end, 1.0, 2)
            line_center = ((line_start[0] + line_end[0])//2, (line_start[1] + line_end[1])//2)
            line_size = np.sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)
            objects.append({"type": "line", "center": line_center, "size": line_size})
            line_placed = True

        # Determine cross orientation based on line
        if is_cross_parallel:
            # One axis of cross is parallel to line
            cross_angle = line_angle
        else:
            # One axis of cross has 45° angle with line
            cross_angle = line_angle + np.pi/4

        # Generate cross with random position that doesn't overlap with other objects
        attempts = 0
        cross_placed = False

        while attempts < 50 and not cross_placed:
            # Longer cross size for collision detection
            cross_size = np.random.uniform(self.img_size * 0.12, self.img_size * 0.18)
            cross_center = (
                np.random.randint(padding, self.img_size - padding),
                np.random.randint(padding, self.img_size - padding)
            )

            # Check distance to previously placed objects
            too_close = False
            for obj in objects:
                dist = np.sqrt((cross_center[0] - obj["center"][0])**2 +
                              (cross_center[1] - obj["center"][1])**2)

                # For the line, calculate distance from point to line instead
                if obj["type"] == "line":
                    # Distance from point to line calculation
                    # Use the perpendicular distance formula
                    continue  # Skip distance check for line

                if dist < min_distance + obj["size"]/2 + cross_size/2:
                    too_close = True
                    break

            if not too_close:
                objects.append({"type": "cross", "center": cross_center, "size": cross_size})
                cross_placed = True
                # Draw cross with thinner and more varied axes
                self._draw_cross_varying_length(img, cross_center, cross_angle, cross_size)

            attempts += 1

        # Determine rectangle orientation based on line
        if is_rectangle_parallel:
            # Long side of rectangle is parallel to line
            rect_angle = line_angle
        else:
            # Long side of rectangle is perpendicular to line
            rect_angle = line_angle + np.pi/2

        # Generate rectangle with random position that doesn't overlap with other objects
        attempts = 0
        rect_placed = False

        while attempts < 50 and not rect_placed:
            # More variable rectangle dimensions
            rect_width = np.random.uniform(self.img_size * 0.05, self.img_size * 0.12)
            rect_height = np.random.uniform(self.img_size * 0.13, self.img_size * 0.22)  # Longer than width
            rect_size = max(rect_width, rect_height)

            rect_center = (
                np.random.randint(padding, self.img_size - padding),
                np.random.randint(padding, self.img_size - padding)
            )

            # Check distance to previously placed objects
            too_close = False
            for obj in objects:
                dist = np.sqrt((rect_center[0] - obj["center"][0])**2 +
                              (rect_center[1] - obj["center"][1])**2)

                # For the line, calculate distance from point to line instead
                if obj["type"] == "line":
                    # Skip distance check for line
                    continue

                if dist < min_distance + obj["size"]/2 + rect_size/2:
                    too_close = True
                    break

            if not too_close:
                objects.append({"type": "rectangle", "center": rect_center, "size": rect_size})
                rect_placed = True
                self._draw_rectangle(img, rect_center, rect_width, rect_height, rect_angle)

            attempts += 1

        return img

    # The rest of the methods remain the same
    def _draw_square(self, img, center_pos, size, angle):
        """Draw a square with the given position, size and rotation angle"""
        x, y = center_pos
        half_size = size / 2

        # Square vertices before rotation (centered at origin)
        square_points = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size)
        ]

        # Apply rotation and translation
        rotated_points = []
        for px, py in square_points:
            # Rotate
            rx = px * np.cos(angle) - py * np.sin(angle)
            ry = px * np.sin(angle) + py * np.cos(angle)
            # Translate
            rx += x
            ry += y
            rotated_points.append((int(rx), int(ry)))

        # Draw filled square
        cv2.fillPoly(img, [np.array(rotated_points, dtype=np.int32)], 1.0)

    def _draw_rectangle(self, img, center_pos, width, height, angle):
        """Draw a thicker rectangle with the given position, width, height, and rotation angle"""
        x, y = center_pos
        half_width = width / 2
        half_height = height / 2

        # Rectangle vertices before rotation (centered at origin)
        rect_points = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]

        # Apply rotation and translation
        rotated_points = []
        for px, py in rect_points:
            # Rotate
            rx = px * np.cos(angle) - py * np.sin(angle)
            ry = px * np.sin(angle) + py * np.cos(angle)
            # Translate
            rx += x
            ry += y
            rotated_points.append((int(rx), int(ry)))

        # Draw filled rectangle (thicker)
        cv2.fillPoly(img, [np.array(rotated_points, dtype=np.int32)], 1.0)

    def _draw_cross_varying_length(self, img, center_pos, angle, max_length=None):
        """Draw a cross with varying axis lengths"""
        x, y = center_pos

        # Random axis lengths with greater variability
        if max_length is None:
            max_length = self.img_size * 0.15

        # Allow for significant difference between axes lengths
        axis1_length = np.random.uniform(max_length * 0.8, max_length * 1.6)
        axis2_length = np.random.uniform(max_length * 0.8, max_length * 1.6)

        # First axis (at specified angle)
        axis1_start = (
            int(x - np.cos(angle) * axis1_length / 2),
            int(y - np.sin(angle) * axis1_length / 2)
        )
        axis1_end = (
            int(x + np.cos(angle) * axis1_length / 2),
            int(y + np.sin(angle) * axis1_length / 2)
        )

        # Second axis (perpendicular to first axis)
        perp_angle = angle + np.pi/2
        axis2_start = (
            int(x - np.cos(perp_angle) * axis2_length / 2),
            int(y - np.sin(perp_angle) * axis2_length / 2)
        )
        axis2_end = (
            int(x + np.cos(perp_angle) * axis2_length / 2),
            int(y + np.sin(perp_angle) * axis2_length / 2)
        )

        # Draw the two lines forming the cross (thinner - width=1)
        cv2.line(img, axis1_start, axis1_end, 1.0, 2)
        cv2.line(img, axis2_start, axis2_end, 1.0, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        # Convert directly to TensorFlow tensor
        img_transformed = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
        # Add channel dimension
        img_transformed = tf.expand_dims(img_transformed, axis=-1)  # TF uses channels-last

        return img_transformed, target

    def visualize_samples(self, num_samples=5):
        """Visualize random samples from each class"""
        fig, axes = plt.subplots(8, num_samples, figsize=(num_samples*2, 16))

        class_descriptions = [
            "Side-aligned line, Parallel cross, Parallel rectangle",
            "Side-aligned line, Angled cross, Parallel rectangle",
            "Side-aligned line, Parallel cross, Perpendicular rectangle",
            "Side-aligned line, Angled cross, Perpendicular rectangle",
            "Diagonal-aligned line, Parallel cross, Parallel rectangle",
            "Diagonal-aligned line, Angled cross, Parallel rectangle",
            "Diagonal-aligned line, Parallel cross, Perpendicular rectangle",
            "Diagonal-aligned line, Angled cross, Perpendicular rectangle"
        ]

        for class_idx in range(8):
            # Get indices for this class
            indices = [i for i, t in enumerate(self.targets) if t == class_idx]
            # Select random samples
            samples = np.random.choice(indices, num_samples, replace=False)

            for i, sample_idx in enumerate(samples):
                img = self.data[sample_idx]
                axes[class_idx, i].imshow(img, cmap='gray')
                if i == 0:  # Add class description to first column
                    axes[class_idx, i].set_title(f"Class {class_idx}", fontsize=8)
                    axes[class_idx, i].set_ylabel(class_descriptions[class_idx], fontsize=7)
                else:
                    axes[class_idx, i].set_title(f"Class {class_idx}")
                axes[class_idx, i].axis('off')

        plt.tight_layout()
        plt.show()


    def plot_class_distribution(self):
        """Plot the distribution of classes in the dataset"""
        plt.figure(figsize=(10, 5))

        # Get targets from the original dataset
        all_targets = self.targets

        class_counts = np.bincount(all_targets)
        plt.bar(range(len(class_counts)), class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(len(class_counts)))
        plt.show()