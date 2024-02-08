"""
Author: Anonymized
Created: Anonymized
Contains overall schema for analytics engine final result
"""

from schema import Schema, And, Or, Optional

analytics_schema = Schema(
    {
        'id': str,
        'keyword': str,

        'metaInfo': {
            'pipelineVersion': Or(str, None),
            'analysisStartTime': int,
            'totalRuntime': float,
            'RunModules': [str],
            'ModuleRuntime': [float],
            'SuccessModules': [str],  # left general for now, need to make it strict like
            'FailureModules': [str],  # Or('audio', 'gaze', 'location', 'posture', None)
        },
        'debugInfo': Or({
            str: object,
        }, None),
        'secondLevel': Or([{
            'secondInfo': {
                'unixSeconds': Or(int, None),
                'frameStart': Or(int, None),
                'frameEnd': Or(int, None),
            },
            'audio': Or({
                'isSilence': Or(True, False, None),
                'isObjectNoise': Or(True, False, None),
                'isTeacherOnly': Or(True, False, None),
                'isSingleSpeaker': Or(True, False, None),
            }, None),
            'gaze': Or({
                'instructor': Or({
                    'angle': Or(float, None),
                    'angleChange': Or(float, None),
                    'direction': Or('left', 'right', None),
                    'viewingSectorThreshold': Or(float, None),
                    'countStudentsInGaze': Or(int, None),
                    'towardsStudents': Or(True, False, None),
                    'lookingDown': Or(True, False, None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'angle': Or([float], None),
                    'angleChange': Or([float], None),
                    'direction': Or([Or('left', 'right')], None),
                    'towardsInstructor': Or([Or(True, False)], None),
                    'lookingDown': Or([Or(True, False)], None),
                    'lookingFront': Or([Or(True, False)], None),
                }, None)
            }, None),
            'location': Or({
                'instructor': Or({
                    'atBoard': Or(True, False, None),
                    'atPodium': Or(True, False, None),
                    'isMoving': Or(True, False, None),
                    'locationCoordinates': Or([int], None),
                    'locationCategory': Or('left', 'right', None),
                    'locationEntropy': Or(float, None),
                    'headEntropy': Or(float, None)
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'trackingIdMap': Or([[int]], None),
                    'isMoving': Or([Or(True, False)], None),
                    'locationCoordinates': Or([[int]], None),
                    'locationCategory': Or([Or('left', 'right')], None),
                    'locationEntropy': Or([float], None),
                    'headEntropy': Or([float], None),
                }, None)
            }, None),
            'posture': Or({
                'instructor': Or({
                    'isStanding': Or(True, False, None),
                    'isPointing': Or(True, False, None),
                    'pointingDirection': Or([float], None),
                    'handPosture': Or(str, None),
                    'headPosture': Or(str, None),
                    'centroidFaceDistance': Or(float, None),
                    'centroidFaceDistanceAbsolute': Or(float, None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'isStanding': Or([Or(True, False)], None),
                    'bodyPosture': Or([str], None),  # sit straight/ sit slouch/ stand
                    'headPosture': Or([str], None),  # looking up or not
                    'handPosture': Or([str], None),  # raised, ontable, onface, down, crossed
                }, None)
            }, None),
            'crossModal': Or({
                str: object,
            }, None)
        }], None),
        'blockLevel': Or([{
            'blockInfo': {
                'unixStartSecond': int,
                'blockLength': int,
                'id': int,
            },
            'audio': Or({
                'silenceFraction': Or(float, None),
                'objectFraction': Or(float, None),
                'teacherOnlyFraction': Or(float, None),
                'singleSpeakerFraction': Or(float, None),
                'teacherActivityType': Or([str], None),
                'teacherActivityFraction': Or([float], None),
                'teacherActivityTimes': Or([[[int]]], None)  # a list like [ [(0,10), (12,14)], [(10,12),(14,20)]]
            }, None),
            'gaze': Or({
                'instructor': Or({
                    'gazeCategory': Or('left', 'right', None),
                    'totalCategoryFraction': Or([float], None),
                    'longestCategoryFraction': Or([float], None),
                    'principalGaze': Or({
                        'direction': Or([float], None),
                        'directionVariation': Or([float], None),
                        'longestStayFraction': Or([[float]], None)
                    }, None),
                    'rollMean': Or(float, None),
                    'yawMean': Or(float, None),
                    'pitchMean': Or(float, None),
                    'rollVariance': Or(float, None),
                    'yawVariance': Or(float, None),
                    'pitchVariance': Or(float, None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'numOccurrencesInBlock': Or([int], None),
                    'gazeCategory': Or('left', 'right', None),
                    'totalCategoryFraction': Or([[float]], None),
                    'longestCategoryFraction': Or([[float]], None),
                    'directionMean': Or([[float]], None),
                    'directionVariation': Or([[float]], None),
                    'towardsInstructorFraction': Or([float], None),
                    'lookingDownFraction': Or([float], None),
                    'lookingFrontFraction': Or([float], None),
                    'rollMean': Or([float], None),
                    'yawMean': Or([float], None),
                    'pitchMean': Or([float], None),
                    'rollVariance': Or([float], None),
                    'yawVariance': Or([float], None),
                    'pitchVariance': Or([float], None),
                }, None)
            }, None),
            'location': Or({
                'instructor': Or({
                    'totalBoardFraction': Or(float, None),
                    'longestBoardFraction': Or(float, None),
                    'totalPodiumFraction': Or(float, None),
                    'longestPodiumFraction': Or(float, None),
                    'totalMovingFraction': Or(float, None),
                    'longestMovingFraction': Or(float, None),
                    'locationCategory': Or([int], None),
                    'CategoryFraction': Or([float], None),
                    'longestCategoryFraction': Or([float], None),
                    'stayAtLocation': Or([[int]], None),
                    'stayAtLocationTimes': Or([[float]], None),
                    'longestStayFraction': Or(float, None),
                    'principalMovement': Or({
                        'directionMean': Or([float], None),
                        'directionVariation': Or([float], None),
                        'directionComps': Or([[float]], None)
                    }, None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'numOccurrencesInBlock': Or([int], None),
                    'isSettled': Or([bool], None),
                    'meanBodyEntropy': Or([float], None),
                    'maxBodyEntropy': Or([float], None),
                    'varBodyEntropy': Or([float], None),
                    'meanHeadEntropy': Or([float], None),
                    'maxHeadEntropy': Or([float], None),
                    'varHeadEntropy': Or([float], None),
                    'stayCoordinates': Or([[int]], None),
                    'clusterCount': Or(int, None),
                    'clusterCenters': Or([[float]], None),
                    'clusterStudentIds': Or([[int]], None),
                    'seatingArrangement': Or(str, None),  # categorical frontFacing/Groups/Circular etc.
                }, None)
            }, None),
            'posture': Or({
                'instructor': Or({
                    'standingFraction': Or(float, None),
                    'handPostureCategory': Or([str], None),
                    'handPostureCategoryFraction': Or([float], None),
                    'headPostureCategory': Or([str], None),
                    'headPostureCategoryFraction': Or([float], None),
                    'meanCentroidFaceDistance': Or(float, None),
                    'varCentroidFaceDistance': Or(float, None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'numOccurrencesInBlock': Or([int], None),
                    'isStandingFraction': Or([Or(True, False)], None),
                    'bodyPostureCategory': Or([str], None),  # [sit straight,sit slouch,stand]
                    'bodyPostureCategoryFraction': Or([[float]], None),  # for each student frac of postures
                    'headPostureCategory': Or([str], None),  # [up, down]
                    'headPostureCategoryFraction': Or([[float]], None),  # for each student frac of postures
                    'handPostureCategory': Or([str], None),
                    'handPostureCategoryFraction': Or([[float]], None),
                }, None)
            }, None),
            'crossModal': Or({
                str: object,
            }, None)

        }], None),
        'sessionLevel': Or({
            'sessionInfo': {
                'unixStartSecond': int,
                'sessionLength': int,
            },
            'audio': Or({
                'audioBasedActivityType': Or([str], None),
                'audioBasedActivityFraction': Or([float], None),
                'audioBasedActivityBlocks': Or([[[int]]], None),
                # a list like [ [(0,10), (12,14)], [(10,12),(14,20)]]
            }, None),
            'gaze': Or({
                'instructor': Or({
                    'gazePreference': Or(str, None),
                    'topLocations': Or([[float]], None),  # get gaze fractions from top locations
                    'topLocationsGazeLeftFraction': Or([[float]], None),  # get gaze left fractions from top locations
                    'objectCategory': Or([str], None),  # can be students, board, book, table
                    'lookingAtObjectFraction': Or([float], None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'gazeCategory': Or([str],None),
                    'gazeCategoryFraction': Or([[float]],None),
                }, None)
            }, None),
            'location': Or({
                'instructor': Or({
                    'locationBasedActivityType': Or([str], None),
                    'locationBasedActivityFraction': Or([float], None),
                    'locationBasedActivityBlocks': Or([[[int]]], None),
                    'locationClusterCount': int,
                    'locationClusterCenter': Or([[float]], None),
                    'locationClusterSize': Or([int], None),  # no. of points in that cluster
                    'totalBoardFraction': Or(float, None),
                    'longestBoardFraction': Or(float, None),
                    'totalPodiumFraction': Or(float, None),
                    'longestPodiumFraction': Or(float, None),
                    'totalMovingFraction': Or(float, None),
                    'longestMovingFraction': Or(float, None),
                    'locationCategory': ('left', 'right'),
                    'CategoryFraction': Or([float], None),
                    'longestCategoryFraction': Or([float], None),
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'settleDownTime': Or([float], None),
                    'entryTime': Or([float], None),
                    'entryLocation': Or([[float]], None),
                    'exitTime': Or([float], None),
                    'exitLocation': Or([[float]], None),
                    'seatingCategories': Or([str], None),
                    'seatingCategoryBlocks': Or([[int]], None),
                }, None)
            }, None),
            'posture': Or({
                'instructor': Or({
                    'bodyPosturePreference': Or(str, None),  # sit/ stand
                    'pointingClusterCount': int,
                    'pointingClusterCenter': Or([[float]], None),
                    'pointingClusterSize': Or([int], None),  # no. of points in that cluster
                }, None),
                'student': Or({
                    'id': Or([int], None),
                    'handPosturePreference': Or([str], None),
                    'headPosturePreference': Or([str], None),
                }, None)
            }, None),
            'crossModal': Or({
                str: object,
            }, None)
        }, None)

    }
)
