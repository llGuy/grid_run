# MADRONA_TRACK_TRACE_SPLIT=0
# HSSD_FIRST_SCENE=0 
# HSSD_NUM_SCENES=16 
# MADRONA_LBVH=0 
# MADRONA_WIDEN=1 
# MADRONA_MWGPU_FORCE_DEBUG=0 
# MADRONA_TIMING_FILE=timing.txt
# MADRONA_RENDER_RESOLUTION=1
# MADRONA_RENDER_MODE=2
# MADRONA_MWGPU_KERNEL_CACHE=cache 
# ./viewer 64 --cuda

import os
import json
from dataclasses import dataclass

RAST_NO = 1
RT_NO = 2

NUM_STEPS = 64
SPLIT_TRACE_DIV = 16

@dataclass
class RenderMode:
    render_no: int
    name: str
    is_rgb: int

# RENDER_MODE_NAME = { RT_NO: "RTWiderDepthOldSync" }

HSSD_BASE_PATH = "madrona_escape_room"
PROCTHOR_BASE_PATH = "madrona_escape_room"
HIDESEEK_BASE_PATH = "gpu_hideseek"
MJX_BASE_PATH = "madrona_mjx"

DO_TRACE_SPLIT = False

BIG_SCENE_NUM_RUNS = 4

CHECK_ALREADY_RUN = 1

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
    texture_cache: str
    render_rgb: int
    hideseek_num_agents: int

class EnvironmentGen:
    def __init__(self, vars: EnvironmentVars):
        self.env_vars = vars

    def generate_str(self):
        command = f"MADRONA_TRACK_TRACE_SPLIT={self.env_vars.track_trace_split} "
        command = command + f"HSSD_FIRST_SCENE={self.env_vars.first_scene_index} "
        command = command + f"HSSD_NUM_SCENES={self.env_vars.num_scenes} "
        command = command + f"MADRONA_LBVH={1} "
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
        command = command + f"MADRONA_TEXTURE_CACHE_DIR={self.env_vars.texture_cache} "
        command = command + f"MADRONA_RENDER_RGB={self.env_vars.render_rgb} "
        command = command + f"HIDESEEK_NUM_AGENTS={self.env_vars.hideseek_num_agents} "

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

def do_procthor_run(rmode, res, num_worlds, num_scenes, num_agents, cache_all=0):
    for i in range(BIG_SCENE_NUM_RUNS):
        output_file_name = f"procthor/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"

        env_vars = EnvironmentVars(
            track_trace_split=0,
            first_scene_index=0,
            num_scenes=num_scenes,
            timing_file=output_file_name,
            render_resolution=res,
            render_mode=rmode.render_no,
            cache_path="procthor_cache",
            bvh_cache_path="procthor_bvh_cache",
            cache_all_bvh=cache_all,
            proc_thor=1,
            seed=i,
            texture_cache="procthor_texture_cache",
            render_rgb=rmode.is_rgb,
            hideseek_num_agents=num_agents
        )

        run_cfg = RunConfig(
            num_worlds=num_worlds,
            num_steps=NUM_STEPS,
            base_path=PROCTHOR_BASE_PATH
        )

        if not os.path.isfile(output_file_name) or not CHECK_ALREADY_RUN:
            print("Haven't performed the run - doing now!")
            hssd_run = HSSDRun(run_cfg, env_vars)
            hssd_run.run()
        else:
            print("Already performed run")

        if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            # Start a split run too
            split_output_file_name = f"procthor/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            env_vars.track_trace_split = 1
            env_vars.timing_file = split_output_file_name
            run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

            if not os.path.isfile(split_output_file_name):
                print("Haven't performed the run - doing now!")
                split_hssd_run = HSSDRun(run_cfg, env_vars)
                split_hssd_run.run()
            else:
                print("Already performed run")

def do_hssd_run(rmode, res, num_worlds, num_scenes, num_agents, cache_all=0):
    for i in range(BIG_SCENE_NUM_RUNS):
        output_file_name = f"hssd/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"

        env_vars = EnvironmentVars(
            track_trace_split=0,
            first_scene_index=0,
            num_scenes=num_scenes,
            timing_file=output_file_name,
            render_resolution=res,
            render_mode=rmode.render_no,
            cache_path="hssd_cache",
            bvh_cache_path="hssd_bvh_cache",
            cache_all_bvh=cache_all,
            proc_thor=0,
            seed=i,
            texture_cache="hssd_texture_cache",
            render_rgb=rmode.is_rgb,
            hideseek_num_agents=num_agents
        )

        run_cfg = RunConfig(
            num_worlds=num_worlds,
            num_steps=NUM_STEPS,
            base_path=HSSD_BASE_PATH
        )

        if not os.path.isfile(output_file_name) or not CHECK_ALREADY_RUN:
            print("Haven't performed the run - doing now!")
            hssd_run = HSSDRun(run_cfg, env_vars)
            hssd_run.run()
        else:
            print("Already performed run")

        if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            # Start a split run too
            split_output_file_name = f"hssd/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            env_vars.track_trace_split = 1
            env_vars.timing_file = split_output_file_name
            run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

            if not os.path.isfile(split_output_file_name):
                print("Haven't performed the run - doing now!")
                split_hssd_run = HSSDRun(run_cfg, env_vars)
                split_hssd_run.run()
            else:
                print("Already performed run")

def do_hideseek_run(rmode, res, num_worlds, num_agents):
    output_file_name = f"hideseek/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_agents}.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=rmode.render_no,
        cache_path="hideseek_cache",
        bvh_cache_path="hideseek_bvh_cache",
        cache_all_bvh=0,
        proc_thor=0,
        seed=0,
        texture_cache="hideseek_texture_cache",
        render_rgb=rmode.is_rgb,
        hideseek_num_agents=num_agents
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=HIDESEEK_BASE_PATH
    )

    hideseek_run = HideseekRun(run_cfg, env_vars)
    hideseek_run.run()

    if rmode.render_no == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"hideseek/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_agents}_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

        split_hideseek_run = HideseekRun(run_cfg, env_vars)
        split_hideseek_run.run()

def do_mjx_run(rmode, res, num_worlds):
    output_file_name = f"mjx/out_{rmode.name}_{num_worlds}_{res}x{res}_1.json"

    env_vars = EnvironmentVars(
        track_trace_split=0,
        first_scene_index=0,
        num_scenes=1,
        timing_file=output_file_name,
        render_resolution=res,
        render_mode=rmode.render_no,
        cache_path="mjx_cache",
        bvh_cache_path="mjx_bvh_cache",
        cache_all_bvh=0,
        proc_thor=0,
        seed=0,
        # No materials here
        texture_cache="mjx_texture_cache",
        render_rgb=0,
        hideseek_num_agents=1
    )

    run_cfg = RunConfig(
        num_worlds=num_worlds,
        num_steps=NUM_STEPS,
        base_path=MJX_BASE_PATH
    )

    mjx_run = MJXRun(run_cfg, env_vars)
    mjx_run.run()

    if rmode.render_no == RT_NO and DO_TRACE_SPLIT:
        # Start a split run too
        split_output_file_name = f"mjx/out_{rmode.name}_{num_worlds}_{res}x{res}_1_split.json"

        env_vars.track_trace_split = 1
        env_vars.timing_file = split_output_file_name
        run_cfg.num_worlds = int(run_cfg.num_worlds / SPLIT_TRACE_DIV)

        split_mjx_run = MJXRun(run_cfg, env_vars)
        split_mjx_run.run()

def calc_hssd_avg(rmode, res, num_worlds, num_scenes, num_agents):
    data = None
    data_split = None

    num_samples = 0

    for i in range(BIG_SCENE_NUM_RUNS):
        output_file_name = f"hssd/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"

        if not os.path.isfile(output_file_name):
            print("File doesn't exit!")
            return

        if data is None:
            with open(output_file_name, "r") as file:
                data = json.load(file)

                num_samples += 1
        else:
            with open(output_file_name, "r") as file:
                new_data = json.load(file)
                new_time = new_data["avg_total_time"]

                if new_time > 1000000.0:
                    print("REJECTED!");
                    continue

                data["avg_total_time"] += new_time
                print(f"got time {new_time}")

                num_samples += 1

                if rmode.render_no == RT_NO:
                    data["avg_trace_time_ratio"] += new_data["avg_trace_time_ratio"]
                    data["render_sort"] += new_data["render_sort"]

        if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            split_output_file_name = f"hssd/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            if data_split is None:
                with open(split_output_file_name, "r") as file:
                    data_split = json.load(file)
            else:
                with open(split_output_file_name, "r") as file:
                    new_data_split = json.load(file)
                    data_split["tlas_percent"] += new_data_split["tlas_percent"]

    data["avg_total_time"] /= float(num_samples)

    if rmode.render_no == RT_NO:
        data["avg_trace_time_ratio"] /= float(num_samples)
        data["render_sort"] /= float(num_samples)

        if DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            data_split["tlas_percent"] /= float(num_samples)

    avg_total_time = data["avg_total_time"]
    print(f"Average total time: {avg_total_time}")

    avg_file_name = f"hssd/avg/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"
    split_avg_file_name = f"hssd/avg/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

    with open(avg_file_name, "w") as file:
        json.dump(data, file, indent=4) 

    if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
        with open(split_avg_file_name, "w") as file:
            json.dump(data_split, file, indent=4) 

def calc_procthor_avg(rmode, res, num_worlds, num_scenes, num_agents):
    data = None
    data_split = None

    num_samples = 0

    for i in range(BIG_SCENE_NUM_RUNS):
        output_file_name = f"procthor/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"

        if data is None:
            with open(output_file_name, "r") as file:
                data = json.load(file)

                num_samples += 1
        else:
            with open(output_file_name, "r") as file:
                new_data = json.load(file)
                new_time = new_data["avg_total_time"]

                if new_time > 1000000.0:
                    print("REJECT!")
                    continue

                data["avg_total_time"] += new_time
                print(f"got time {new_time}")

                num_samples += 1

                if rmode.render_no == RT_NO:
                    data["avg_trace_time_ratio"] += new_data["avg_trace_time_ratio"]
                    data["render_sort"] += new_data["render_sort"]

        if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            split_output_file_name = f"procthor/run{i}/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

            if data_split is None:
                with open(split_output_file_name, "r") as file:
                    data_split = json.load(file)
            else:
                with open(split_output_file_name, "r") as file:
                    new_data_split = json.load(file)
                    data_split["tlas_percent"] += new_data_split["tlas_percent"]

    data["avg_total_time"] /= float(num_samples)

    if rmode.render_no == RT_NO:
        data["avg_trace_time_ratio"] /= float(num_samples)
        data["render_sort"] /= float(num_samples)

        if DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
            data_split["tlas_percent"] /= float(num_samples)

    avg_total_time = data["avg_total_time"]
    print(f"Average total time: {avg_total_time}")

    avg_file_name = f"procthor/avg/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_{num_agents}.json"
    split_avg_file_name = f"procthor/avg/out_{rmode.name}_{num_worlds}_{res}x{res}_{num_scenes}_split.json"

    with open(avg_file_name, "w") as file:
        json.dump(data, file, indent=4) 

    if rmode.render_no == RT_NO and DO_TRACE_SPLIT and num_worlds / SPLIT_TRACE_DIV >= num_scenes:
        with open(split_avg_file_name, "w") as file:
            json.dump(data_split, file, indent=4) 

def test():
    render_modes_list = [ 
        RenderMode(render_no=RT_NO, name="test", is_rgb=1)
    ]

    render_resolutions_list = [ 64 ]
    num_worlds_list = [ 64 ]
    num_unique_scenes_list = [ 16 ]

    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                do_hssd_run(render_mode, render_resolution, 
                        num_worlds, 16, 8);


def do_multi_view_scaling():
    render_modes_list = [ 
        RenderMode(render_no=RT_NO, name="RTColor", is_rgb=1),
        RenderMode(render_no=RAST_NO, name="RastColor", is_rgb=1),
    ]

    render_resolutions_list = [ 64 ]
    num_worlds_list = [ 2048 ]

    num_agents_list = [1,2, 4, 8, 16,32]

    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_agents in num_agents_list:
                    do_hideseek_run(render_mode, render_resolution, 
                            num_worlds, num_agents);


# Perform the runs
def main():
    # cache_hssd_bvh()
    # cache_procthor_bvh()

    # Let's first cache all the BVHs for HSSD
    render_modes_list = [ 
        RenderMode(render_no=RAST_NO, name="RastColor", is_rgb=1),
        RenderMode(render_no=RT_NO, name="RTColor", is_rgb=1),
    ]

    render_resolutions_list = [ 64, 128, 256 ]
    num_worlds_list = [ 1024 ]
    num_unique_scenes_list = [ 16 ]

    num_agents_list = [1]

    """
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    for num_agents in num_agents_list:
                        do_procthor_run(render_mode, render_resolution, 
                                num_worlds, num_unique_scenes, num_agents);
    """

    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                do_hideseek_run(render_mode, render_resolution, 
                        num_worlds, 5);

    """
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                do_mjx_run(render_mode, render_resolution, 
                        num_worlds);
    """

    """
    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    for num_agents in num_agents_list:
                        do_procthor_run(render_mode, render_resolution, 
                                num_worlds, num_unique_scenes, num_agents);
                        calc_procthor_avg(render_mode, render_resolution, 
                                num_worlds, num_unique_scenes, num_agents);

    for render_mode in render_modes_list:
        for render_resolution in render_resolutions_list:
            for num_worlds in num_worlds_list:
                for num_unique_scenes in num_unique_scenes_list:
                    for num_agents in num_agents_list:
                        do_hssd_run(render_mode, render_resolution, 
                                num_worlds, num_unique_scenes, num_agents);
                        calc_hssd_avg(render_mode, render_resolution, 
                                num_worlds, num_unique_scenes, num_agents);
    """


if __name__ == "__main__":
    # test()
    main()
    # do_multi_view_scaling()
