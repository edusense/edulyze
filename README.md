# Edulyze: Learning Analytics for Real-World Classrooms at Scale
An analytics system that combines data from various classroom sensing systems to generate analytical insights for relevant pedagogical questions and presents a unified schema to structure processed classroom data.

[[paper (Journal of Learning Analytics 2024)](https://www.learning-analytics.info/index.php/JLA/article/view/8367)]

**Abstract:**
Classroom sensing systems can capture data on teacher-student behaviours and interactions at a scale far greater than human observers can. These data, translated to multi-modal analytics, can provide meaningful insights to educational stakeholders. However, complex data can be difficult to make sense of. In addition, analyses done on these data are often limited by the organization of the underlying sensing system, and translating sensing data into meaningful insights often requires custom analyses across different modalities. We present Edulyze, an analytics engine that processes complex, multi-modal sensing data and translates them into a unified schema that is agnostic to the underlying sensing system or classroom configuration. We evaluate Edulyze’s performance by integrating three sensing systems (Edusense, ClassGaze, and Moodoo) and then present data analyses of five case studies of relevant pedagogical research questions across these sensing systems. We demonstrate how Edulyze’s flexibility and customizability allow us to answer a broad range of research questions made possible by Edulyze’s translation of a breadth of raw sensing data from different sensing systems into relevant classroom analytics. 

## Reference

```bibtex
@article{Patidar_Ngoon_Vogety_Behari_Harrison_Zimmerman_Ogan_Agarwal_2024,
	title        = {Edulyze: Learning Analytics for Real-World Classrooms at Scale},
	author       = {Patidar, Prasoon and Ngoon, Tricia and Vogety, Neeharika and Behari, Nikhil and Harrison, Chris and Zimmerman, John and Ogan, Amy and Agarwal, Yuvraj},
	year         = 2024,
	month        = {Aug.},
	journal      = {Journal of Learning Analytics},
	volume       = 11,
	number       = 2,
	pages        = {297--313},
	doi          = {10.18608/jla.2024.8367},
	url          = {https://www.learning-analytics.info/index.php/JLA/article/view/8367},
}
```

## Installation and Running the Edulyze System
1. Clone the repository, and go to main directory.
2. Create a conda environment using `conda create -n edulyze python=3.9`
3. Activate the environment using `conda activate edulyze`
4. Install the required packages using `pip install -r requirements.txt`
5. Run the service using `python run_edulyze.py --run_config <config_file_path>`

## Configuration
The system can be configured using a configuration file. The configuration file is a JSON file that contains the following fields:

- `driver`: The driver for underlying sensing systems. Currently, the system supports the following drivers:
  - `edusense`: A driver for the EduSense classroom sensing system.
  - `classgaze`: A driver for the ClassGaze classroom sensing system. This is same as edusense driver. In latest version, both sensing systems are merged in one edusense driver.
  - `moodoo`: A driver for the Moodoo classroom sensing system.
- `run_mode`: can be `single`/`multiple` based on if we want to process a single session or batch process multiple sessions.
- `input_data_type`: The name of the data source. Currently, we are using `graphql` as a data source for edusense and classgaze, and `file` for moodoo.
- IF run_mode is `multiple`:
  - `session_list_filepath`: The path to the file containing the list of sessions to be processed.
- IF run_mode is `single`:
  - `session_id`: The session id of the session to be processed.
  - `session_keyword`: The keyword assignment of the sessions to be processed. This should be of format `<course-code>_<physical-classroom-location>_<semester>_<session-date>_<session-start-time>`.
- DATA Fetch Configs: These are the configurations for fetching data from a mongodb data source supported with graphql.
  - `server_name`: A custom name assigned to create local cache.
  - `server_user`: The database name of the mongodb database.
  - `server_password`: The database password for authentication purposes.
  - `server_backend_url`: The complete url with port of where graphql endpoint is deployed.
  - `audio_server_name[depreciated]`: Was used if audio information is stored in a different server.
- OUTPUT WRITE CONFIGS: These are the configurations for writing the output to a graphql end point and files as a unified edulyze schema, which can be used further by applications to extract data
  - `output_prefix`: A custom name assigned as a file prefix for output files.
  - `output_server_name`: A custom name assigned to create local cache.
  - `output_server_user`: The database name of the mongodb database.
  - `output_server_password`: The database password for authentication purposes.
  - `output_server_url`: The complete url with port of where graphql endpoint is deployed.
- LOGGING CONFIGS: These are the configurations for logging the output of the system.
  - `log_dir`: The directory where the logs will be stored.
  - `log_file_prefic`: The prefix of the log file.
- CACHE/DEBUG CONFIGS: These are the configurations for running the system in different caching/debug mode.
  - `debug_mode`: A boolean flag to enable debug mode.
  - `cache_mode`: A boolean flag to enable caching of data.
  - `cache_update`: A boolean flag to enable updating the cache.
  - `cache_dir`: The directory where the cache will be stored.

## Example Configuration

Example configuration files for different drivers can be found in the `example_configs` directory.
1. `edusense_classgaze_multiple.json`: An example configuration file for processing multiple sessions from EduSense and ClassGaze.
2. `edusense_classgaze_single.json`: An example configuration file for processing a single session from EduSense and ClassGaze.
3. `moodoo.json`: An example configuration file for processing sessions from cache files from Moodoo.
4. `sample_session_info.csv`: An example file containing the list of sessions to be processed if run mode is `multiple`.

## Driver Interface

The driver interface is a set of functions that the system uses to interact with the underlying sensing systems. The driver interface is defined in the `drivers/DriverInterface.py` file. The driver interface contains the following functions:

1. `getAudioInput`: A function to get the audio input from the underlying sensing system in edulyze format.
2. `getLocationInput`: A function to get the location input from the underlying sensing system in edulyze format.
3. `getGazeInput`: A function to get the gaze input from the underlying sensing system in edulyze format.
4. `getMetaInput`: A function to get the meta input, like location annotation etc. from the underlying sensing system in edulyze format.

For more information, look at drivers created for edusense, classgaze and moodoo.

## Output Schema definition

The output schema is a unified schema that structures the processed classroom data. The output schema is defined in the `post_analytics/init_empty_output_schema.py` file. More information about the output schema can be found in the edulyze paper. Here is the structure of the output schema:

```python
{
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
    'secondLevel': None, # look below for more details on schema of second info
    'blockLevel': None, # look below for more details on schema of block info
    'sessionLevel': None # look below for more details on schema of session info
}
```
<details>
<summary>Second Level Schema</summary>


The `secondLevel` field contains the following fields:

```python
{
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
```

</details>

<details>
<summary>Block Level Schema</summary>


The `blockLevel` field contains the following fields:

```python
{
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
```
</details>

<details>
<summary>Session Level Schema</summary>

The `sessionLevel` field consist of following schema.

```python
{
    'sessionInfo': {
        'unixStartSecond': int,
        'sessionLength': int,
    },
    'audio': {
        'audioBasedActivityType': None,
        'audioBasedActivityFraction': None,
        'audioBasedActivityBlocks': None, # a list like [ [(0,10), (12,14)], [(10,12),(14,20)]]
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
```
</details>

The schema is generated using post analytics output automatically.

## Additional Information

For more information, please refer to the Edulyze paper. If you have any questions, please contact the authors of the paper.






