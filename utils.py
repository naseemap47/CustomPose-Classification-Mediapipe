import numpy as np


def get_pose_center(landmark_names, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[landmark_names.index('left_hip')]
    right_hip = landmarks[landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center


def get_pose_size(landmark_names, landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """

    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    left_hip = landmarks[landmark_names.index('left_hip')]
    right_hip = landmarks[landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = get_pose_center(landmark_names, landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)


def NormPoseLandmark(landmark_names, landmarks):

    # Normalize translation.
    pose_center = get_pose_center(landmark_names, landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = get_pose_size(
        landmark_names, landmarks, torso_size_multiplier=2.5)
    landmarks /= pose_size

    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks
