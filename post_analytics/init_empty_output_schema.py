"""
Author: Anonymized
Created: Anonymized
Empty analytics engine output payload including second level, block level and session level schema
"""

from schema import Schema, And, Or, Optional
from copy import deepcopy

analytics_schema = {
    'id': None,
    'keyword': None,

    'metaInfo': {
        'pipelineVersion': None,
        'analysisStartTime': None,
        'totalRuntime': None,
        'RunModules': None,
        'ModuleRuntime': None,
        'SuccessModules': None,
        'FailureModules': None,
    },
    'debugInfo': None,
    'secondLevel': None,
    'blockLevel': None,
    'sessionLevel': None
}

second_empty_schema = {
    'secondInfo': {
        'unixSeconds': None,
        'frameStart': None,
        'frameEnd': None,
    },
    'audio': {
        'isSilence': None,
        'isObjectNoise': None,
        'isTeacherOnly': None,
        'isSingleSpeaker': None,
    },
    'gaze': {
        'instructor': {
            'angle': None,
            'angleChange': None,
            'direction': None,
            'viewingSectorThreshold': None,
            'countStudentsInGaze': None,
            'towardsStudents': None,
            'lookingDown': None,
        },
        'student': {
            'id': None,
            'angle': None,
            'angleChange': None,
            'direction': None,
            'towardsInstructor': None,
            'lookingDown': None,
            'lookingFront': None,
        }
    },
    'location': {
        'instructor': {
            'atBoard': None,
            'atPodium': None,
            'isMoving': None,
            'locationCoordinates': None,
            'locationCategory': None,
            'locationEntropy': None,
            'headEntropy': None,
        },
        'student': {
            'id': None,
            'trackingIdMap': None,
            'isMoving': None,
            'locationCoordinates': None,
            'locationCategory': None,
            'locationEntropy': None,
            'headEntropy': None,
        }
    },
    'posture': {
        'instructor': {
            'isStanding': None,
            'isPointing': None,
            'pointingDirection': None,
            'handPosture': None,
            'headPosture': None,
            'centroidFaceDistance': None,
            'centroidFaceDistanceAbsolute': None,
        },
        'student': {
            'id': None,
            'isStanding': None,
            'bodyPosture': None,
            'headPosture': None,
            'handPosture': None,
        }
    },
    'crossModal': None
}
block_empty_schema = {
    'blockInfo': {
        'unixStartSecond': int,
        'blockLength': int,
        'id': int,
    },
    'audio': {
        'silenceFraction': None,
        'objectFraction': None,
        'teacherOnlyFraction': None,
        'singleSpeakerFraction': None,
        'teacherActivityType': None,
        'teacherActivityFraction': None,
        'teacherActivityTimes': None,
    },
    'gaze': {
        'instructor': {
            'gazeCategory': None,
            'totalCategoryFraction': None,
            'longestCategoryFraction': None,
            'principalGaze': {
                'direction': None,
                'directionVariation': None,
                'longestStayFraction': None,
            },
            'rollMean': None,
            'yawMean': None,
            'pitchMean': None,
            'rollVariance': None,
            'yawVariance': None,
            'pitchVariance': None,
        },
        'student': {
            'id': None,
            'numOccurrencesInBlock': None,
            'gazeCategory': None,
            'totalCategoryFraction': None,
            'longestCategoryFraction': None,
            'directionMean': None,
            'directionVariation': None,
            'towardsInstructorFraction': None,
            'lookingDownFraction': None,
            'lookingFrontFraction': None,
            'rollMean': None,
            'yawMean': None,
            'pitchMean': None,
            'rollVariance': None,
            'yawVariance': None,
            'pitchVariance': None,
        }
    },
    'location': {
        'instructor': {
            'totalBoardFraction': None,
            'longestBoardFraction': None,
            'totalPodiumFraction': None,
            'longestPodiumFraction': None,
            'totalMovingFraction': None,
            'longestMovingFraction': None,
            'locationCategory': ('left', 'right'),
            'CategoryFraction': None,
            'longestCategoryFraction': None,
            'stayAtLocation': None,
            'stayAtLocationTimes': None,
            'longestStayFraction': None,
            'principalMovement': {
                'directionMean': None,
                'directionVariation': None,
                'directionComps': None,
            },
        },
        'student': {
            'id': None,
            'numOccurrencesInBlock': None,
            'isSettled': None,
            'meanBodyEntropy': None,
            'maxBodyEntropy': None,
            'varBodyEntropy': None,
            'meanHeadEntropy': None,
            'maxHeadEntropy': None,
            'varHeadEntropy': None,
            'stayCoordinates': None,
            'clusterCount': None,
            'clusterCenters': None,
            'clusterStudentIds': None,
            'seatingArrangement': None,
        }
    },
    'posture': {
        'instructor': {
            'standingFraction': None,
            'handPostureCategory': None,
            'handPostureCategoryFraction': None,
            'headPostureCategory': None,
            'headPostureCategoryFraction': None,
            'meanCentroidFaceDistance': None,
            'varCentroidFaceDistance': None,
        },
        'student': {
            'id': None,
            'numOccurrencesInBlock': None,
            'isStandingFraction': None,
            'bodyPostureCategory': None,
            'bodyPostureCategoryFraction': None,
            'headPostureCategory': None,
            'headPostureCategoryFraction': None,
            'handPostureCategory': None,
            'handPostureCategoryFraction': None,
        }
    },
    'crossModal': None

}
session_empty_schema = {
    'sessionInfo': {
        'unixStartSecond': int,
        'sessionLength': int,
    },
    'audio': {
        'audioBasedActivityType': None,
        'audioBasedActivityFraction': None,
        'audioBasedActivityBlocks': None,
        # a list like [ [(0,10), (12,14)], [(10,12),(14,20)]]
    },
    'gaze': {
        'instructor': {
            'gazePreference': None,
            'topLocations': None,
            'topLocationsGazeLeftFraction': None,
            'objectCategory': None,
            'lookingAtObjectFraction': None,
        },
        'student': {
            'id': None,
            'gazeCategory': None,
            'gazeCategoryFraction': None,
        }
    },
    'location': {
        'instructor': {
            'locationBasedActivityType': None,
            'locationBasedActivityFraction': None,
            'locationBasedActivityBlocks': None,
            'locationClusterCount': int,
            'locationClusterCenter': None,
            'locationClusterSize': None,
            'totalBoardFraction': None,
            'longestBoardFraction': None,
            'totalPodiumFraction': None,
            'longestPodiumFraction': None,
            'totalMovingFraction': None,
            'longestMovingFraction': None,
            'locationCategory': ('left', 'right'),
            'CategoryFraction': None,
            'longestCategoryFraction': None,
        },
        'student': {
            'id': None,
            'settleDownTime': None,
            'entryTime': None,
            'entryLocation': None,
            'exitTime': None,
            'exitLocation': None,
            'seatingCategories': None,
            'seatingCategoryBlocks': None,
        }
    },
    'posture': {
        'instructor': {
            'bodyPosturePreference': None,
            'pointingClusterCount': int,
            'pointingClusterCenter': None,
            'pointingClusterSize': None,
        },
        'student': {
            'id': None,
            'handPosturePreference': None,
            'headPosturePreference': None,
        }
    },
    'crossModal': None
}


def init_empty_schema():
    """Returns overall empty schema"""
    return deepcopy(analytics_schema)


def get_empty_second_schema():
    """Returns empty schema for second level analytics"""
    return deepcopy(second_empty_schema)


def get_empty_block_schema():
    """Returns empty schema for block level analytics"""
    return deepcopy(block_empty_schema)


def get_empty_session_schema():
    """Returns empty schema for session level analytics"""
    return deepcopy(session_empty_schema)


