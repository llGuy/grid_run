import os
from dataclasses import dataclass

# The name that you enter here, will be outputted in the JSON file name
# after a run.
RENDER_MODE_NAME = { RAST_NO: "RastBPS" }

# Numbers which get passed as environment variables
RAST_NO = 1
RT_NO = 2

# Configure number of steps
NUM_STEPS = 32

SPLIT_TRACE_DIV = 16

HSSD_BASE_PATH = "madrona_escape_room"
PROCTHOR_BASE_PATH = "madrona_escape_room"
HIDESEEK_BASE_PATH = "gpu_hideseek"
MJX_BASE_PATH = "madrona_mjx"

# Do we do a separate run for gathering info about how much time was spent
# in BLAS or TLAS.
DO_TRACE_SPLIT = False

# Environment variables that all runs will set (although not all need)
@dataclass
class EnvironmentVars:
    track_trace_split: int
    first_scene_index: int
    num_scenes: int
    timing_file: str
    render_resolution: int
    render_mode: int
    cache_path: str
    bvh_cache_path: str
    cache_all_bvh: int
    proc_thor: int
    seed: int

class EnvironmentGen:
    def __init__(self, vars: EnvironmentVars):
        self.env_vars = vars

    def generate_str(self):
        command = f"MADRONA_TRACK_TRACE_SPLIT={self.env_vars.track_trace_split} "
        command = command + f"HSSD_FIRST_SCENE={self.env_vars.first_scene_index} "
        command = command + f"HSSD_NUM_SCENES={self.env_vars.num_scenes} "
        command = command + f"MADRONA_LBVH={0} "
        command = command + f"MADRONA_WIDEN={1} "
        command = command + f"MADRONA_MWGPU_FORCE_DEBUG={0} "
        command = command + f"MADRONA_TIMING_FILE={self.env_vars.timing_file} "
        command = command + f"MADRONA_RENDER_RESOLUTION={self.env_vars.render_resolution} "
        command = command + f"MADRONA_RENDER_MODE={self.env_vars.render_mode} "
        command = command + f"MADRONA_MWGPU_KERNEL_CACHE={self.env_vars.cache_path} "
        command = command + f"MADRONA_CACHE_ALL_BVH={self.env_vars.cache_all_bvh} "
        command = command + f"MADRONA_BVH_CACHE_DIR={self.env_vars.bvh_cache_path} "
        command = command + f"MADRONA_PROC_THOR={self.env_vars.proc_thor} "
        command = command + f"MADRONA_SEED={self.env_vars.seed} "

        return command



# Configurations of the various specific runs
@dataclass
class RunConfig:
    num_worlds: int
    num_steps: int
    base_path: str

class MJXRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        script_path = self.run_cfg.base_path + "/scripts/headless.py"

        env_gen = EnvironmentGen(self.env_vars)

        command = env_gen.generate_str()

        command = command + f"python {script_path} "
        command = command + f"--num-worlds {self.run_cfg.num_worlds} "
        command = command + f"--num-steps {self.run_cfg.num_steps} "
        command = command + f"--batch-render-view-width {self.env_vars.render_resolution} "
        command = command + f"--batch-render-view-height {self.env_vars.render_resolution}"

        print(command)
        os.system(command)

class HSSDRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        bin_path = self.run_cfg.base_path + f"/build/headless"

        env_gen = EnvironmentGen(self.env_vars)

        command = env_gen.generate_str()

        command = command + f"{bin_path} "
        command = command + f"CUDA "
        command = command + f"{self.run_cfg.num_worlds} "
        command = command + f"{self.run_cfg.num_steps} "

        print(command)
        os.system(command)

class HideseekRun:
    def __init__(self, run_cfg: RunConfig, env_vars: EnvironmentVars):
        self.run_cfg = run_cfg
        self.env_vars = env_vars

    def run(self):
        bin_path = self.run_cfg.base_path + f"/build/headless"

        env_gen = EnvironmentGen(self.env_vars)
        
        command = env_gen.generate_str()
        command = command + f"{bin_path} "
        command = command + f"CUDA "
        command = command + f"{self.run_cfg.num_worlds} "
        command = command + f"{self.run_cfg.num_steps}"

        print(command)
        os.system(command)

def cache_procthor_bvh():
    output_file_name = f"dummy"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=32,
        render_mode=RT_NO,
        cache_path="procthor_cache",
        bvh_cache_path="procthor_bvh_cache",
        cache_all_bvh=1,
        proc_thor=1,
        seed=0
    )

    run_cfg = RunConfig(
        num_worlds=512,
        num_steps=NUM_STEPS,
        base_path=PROCTHOR_BASE_PATH
    )

    hssd_run = HSSDRun(run_cfg, env_vars)
    hssd_run.run()

def cache_hssd_bvh():
    output_file_name = f"dummy"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=32,
        render_mode=RT_NO,
        cache_path="hssd_cache",
        bvh_cache_path="hssd_bvh_cache",
        cache_all_bvh=1,
        proc_thor=0,
        seed=0
    )

    run_cfg = RunConfig(
        num_worlds=512,
        num_steps=NUM_STEPS,
        base_path=HSSD_BASE_PATH
    )

    hssd_run = HSSDRun(run_cfg, env_vars)
    hssd_run.run()

def do_procthor_run(render_mode, res, num_worlds, num_scenes, cache_all=0):
    for i in range(8):
        output_file_name = f"procthor/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"

        env_vars = EnvironmentVars(
            track_trace_split=0,
            first_scene_index=0,
            num_scenes=num_scenes,
            timing_file=output_file_name,
            render_resolution=res,
            render_mode=render_mode,
            cache_path="procthor_cache",
            bvh_cache_path="procthor_bvh_cache",
            cache_all_bvh=cache_all,
            proc_thor=1,
            seed=i
        )

        run_cfg = RunConfig(
            num_worlds=num_worlds,
            num_steps=NUM_STEPS,
            base_path=PROCTHOR_BASE_PATH
        )

        if not os.path.isfile(output_file_name):
            print("Haven't performed the run - doing now!")
            hssd_run = HSSDRun(run_cfg, env_vars)
            hssd_run.run()
        else:
            print("Already performed run")

        if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            # Start a split run too
            split_output_file_name = f"procthor/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            env_vars.track_trace_split = 1
            env_vars.timing_file = split_output_file_name
            run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

            if not os.path.isfile(split_output_file_name):
                print("Haven't performed the run - doing now!")
                split_hssd_run = HSSDRun(run_cfg, env_vars)
                split_hssd_run.run()
            else:
                print("Already performed run")

def do_hssd_run(render_mode, res, num_worlds, num_scenes, cache_all=0):
    for i in range(8):
        output_file_name = f"hssd/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"

        env_vars = EnvironmentVars(
            track_trace_split=0,
            first_scene_index=0,
            num_scenes=num_scenes,
            timing_file=output_file_name,
            render_resolution=res,
            render_mode=render_mode,
            cache_path="hssd_cache",
            bvh_cache_path="hssd_bvh_cache",
            cache_all_bvh=cache_all,
            proc_thor=0,
            seed=i
        )

        run_cfg = RunConfig(
            num_worlds=num_worlds,
            num_steps=NUM_STEPS,
            base_path=HSSD_BASE_PATH
        )

        if not os.path.isfile(output_file_name):
            print("Haven't performed the run - doing now!")
            hssd_run = HSSDRun(run_cfg, env_vars)
            hssd_run.run()
        else:
            print("Already performed run")

        if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            # Start a split run too
            split_output_file_name = f"hssd/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            env_vars.track_trace_split = 1
            env_vars.timing_file = split_output_file_name
            run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

            if not os.path.isfile(split_output_file_name):
                print("Haven't performed the run - doing now!")
                split_hssd_run = HSSDRun(run_cfg, env_vars)
                split_hssd_run.run()
            else:
                print("Already performed run")

def calc_procthor_avg(render_mode, res, num_worlds, num_scenes):
    data = None
    data_split = None

    for i in range(8):
        output_file_name = f"procthor/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"

        if data is None:
            with open(output_file_name, "r") as file:
                data = json.load(file)
        else:
            with open(output_file_name, "r") as file:
                new_data = json.load(file)
                new_time = new_data["avg_total_time"]

                data["avg_total_time"] += new_time
                print(f"got time {new_time}")

                if render_mode == RT_NO:
                    data["avg_trace_time_ratio"] += new_data["avg_trace_time_ratio"]

        if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            split_output_file_name = f"procthor/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            if data_split is None:
                with open(split_output_file_name, "r") as file:
                    data_split = json.load(file)
            else:
                with open(split_output_file_name, "r") as file:
                    new_data_split = json.load(file)
                    data_split["tlas_percent"] += new_data_split["tlas_percent"]

    data["avg_total_time"] /= 8.0

    if render_mode == RT_NO:
        data["avg_trace_time_ratio"] /= 8.0

        if DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            data_split["tlas_percent"] /= 8.0

    avg_total_time = data["avg_total_time"]
    print(f"Average total time: {avg_total_time}")

    avg_file_name = f"procthor/avg/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"
    split_avg_file_name = f"procthor/avg/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

    with open(avg_file_name, "w") as file:
        json.dump(data, file, indent=4) 

    if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
        with open(split_avg_file_name, "w") as file:
            json.dump(data_split, file, indent=4) 

def calc_hssd_avg(render_mode, res, num_worlds, num_scenes):
    data = None
    data_split = None

    for i in range(8):
        output_file_name = f"hssd/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"

        if data is None:
            with open(output_file_name, "r") as file:
                data = json.load(file)
        else:
            with open(output_file_name, "r") as file:
                new_data = json.load(file)
                new_time = new_data["avg_total_time"]

                data["avg_total_time"] += new_time
                print(f"got time {new_time}")

                if render_mode == RT_NO:
                    data["avg_trace_time_ratio"] += new_data["avg_trace_time_ratio"]

        if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            split_output_file_name = f"hssd/run{i}/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            if data_split is None:
                with open(split_output_file_name, "r") as file:
                    data_split = json.load(file)
            else:
                with open(split_output_file_name, "r") as file:
                    new_data_split = json.load(file)
                    data_split["tlas_percent"] += new_data_split["tlas_percent"]

    data["avg_total_time"] /= 8.0

    if render_mode == RT_NO:
        data["avg_trace_time_ratio"] /= 8.0

        if DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            data_split["tlas_percent"] /= 8.0

    avg_total_time = data["avg_total_time"]
    print(f"Average total time: {avg_total_time}")

    avg_file_name = f"hssd/avg/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}.json"
    split_avg_file_name = f"hssd/avg/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

    with open(avg_file_name, "w") as file:
        json.dump(data, file, indent=4) 

    if render_mode == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
        with open(split_avg_file_name, "w") as file:
            json.dump(data_split, file, indent=4) 

def do_hideseek_run(render_mode, res, num_worlds):
    output_file_name = f"hideseek/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=render_mode,
        cache_path="hideseek_cache",
        bvh_cache_path="hideseek_bvh_cache",
        cache_all_bvh=0,
        proc_thor=0,
        seed=0
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=HIDESEEK_BASE_PATH
    )

    hideseek_run = HideseekRun(run_cfg, env_vars)
    hideseek_run.run()

    if render_mode == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"hideseek/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

        split_hideseek_run = HideseekRun(run_cfg, env_vars)
        split_hideseek_run.run()

def do_mjx_run(render_mode, res, num_worlds):
    output_file_name = f"mjx/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=render_mode,
        cache_path="mjx_cache",
        bvh_cache_path="mjx_bvh_cache",
        cache_all_bvh=0,
        proc_thor=0,
        seed=0
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=MJX_BASE_PATH
    )

    mjx_run = MJXRun(run_cfg, env_vars)
    mjx_run.run()

    if render_mode == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"mjx/out_{RENDER_MODE_NAME[render_mode]}_{num_worlds}_{res}x{res}_1_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

        split_mjx_run = MJXRun(run_cfg, env_vars)
        split_mjx_run.run()


# Perform the runs
def main():
    # Let's first cache all the BVHs for HSSD
    render_modes_list = [ RAST_NO ]
    render_resolutions_list = [ 32, 64, 128 ]
    num_worlds_list = [ 512, 1024, 2048 ]
    num_unique_scenes_list = [ 16, 32, 64 ]

    # Run the HSSD environments:
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    do_hssd_run(render_mode, render_resolution, 
                            num_worlds, num_unique_scenes);
                    calc_hssd_avg(render_mode, render_resolution,
                            num_worlds, num_unique_scenes)

    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    do_procthor_run(render_mode, render_resolution, 
                            num_worlds, num_unique_scenes);
                    calc_procthor_avg(render_mode, render_resolution,
                            num_worlds, num_unique_scenes)

if __name__ == "__main__":
    main()
