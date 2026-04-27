"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import math
import os

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.layout_utils import layouts as layout
from scalesim.single_layer_sim import single_layer_sim as layer_sim
from scalesim.linear_model.tpu import tpuv4_linear_model, tpuv5e_linear_model, tpuv6e_linear_model
from scalesim.linear_model.gpu import ga10b_linear_model


class simulator:
    """
    Class which runs the simulations and manages generated data across various layers
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.conf = cfg()
        self.topo = topo()
        self.layout = layout()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []

        # GPU-mode analytical results (populated when GPU mode is active)
        self.gpu_layer_results = []

        self.params_set_flag = False
        self.all_layer_run_done = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   layout_obj=layout(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True
                   ):
        """
        Method to set the run parameters including inputs and parameters for housekeeping.
        """
        self.conf = config_obj
        self.topo = topo_obj
        self.layout = layout_obj

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True

    #
    def run(self):
        """
        Method to run scalesim simulation for all layers. This method first runs compute and memory
        simulations for each layer and gathers the required stats. Once the simulation runs are
        done, it gathers the stats from single_layer_sim objects and calls generate_report() method
        to create the report files. If save_trace flag is set, then layer wise traces are saved as
        well.
        """
        assert self.params_set_flag, 'Simulator parameters are not set'

        if not os.path.isdir(self.top_path):
            os.mkdir(self.top_path)

        report_path = self.top_path + '/' + self.conf.get_run_name()

        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path

        self._run_standard()

        self.all_layer_run_done = True
        self.generate_reports()

    #
    def _run_standard(self):
        """
        Standard SCALE-Sim simulation path: runs full cycle-accurate memory trace
        simulation for each layer using single_layer_sim objects.
        """
        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            this_layer_sim.set_params(layer_id=i,
                                 config_obj=self.conf,
                                 topology_obj=self.topo,
                                 layout_obj=self.layout,
                                 verbose=self.verbose)
            
            # Prevent OOM in GPU mode by forcing ESTIMATE BANDWIDTH mode (disables massive traces)
            if self.conf.is_gpu_mode():
                this_layer_sim.config.use_user_dram_bandwidth = lambda: False

            self.single_layer_sim_object_list.append(this_layer_sim)

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list:

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
                print('\nRunning Layer ' + str(layer_id))

            single_layer_obj.run()

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                total_cycles = comp_items[0]
                comp_cycles = comp_items[1]
                stall_cycles = comp_items[2]
                util = comp_items[3]
                mapping_eff = comp_items[4]
                print('Total cycles: ' + str(total_cycles))
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                if self.conf.sparsity_support is True:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_filter_metadata_sram_bw = avg_bw_items[2]
                    avg_ofmap_sram_bw = avg_bw_items[3]
                    avg_ifmap_dram_bw = avg_bw_items[4]
                    avg_filter_dram_bw = avg_bw_items[5]
                    avg_ofmap_dram_bw = avg_bw_items[6]
                else:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_ofmap_sram_bw = avg_bw_items[2]
                    avg_ifmap_dram_bw = avg_bw_items[3]
                    avg_filter_dram_bw = avg_bw_items[4]
                    avg_ofmap_dram_bw = avg_bw_items[5]

                print('Average IFMAP SRAM BW: ' + "{:.3f}".format(avg_ifmap_sram_bw) + \
                      ' words/cycle')
                print('Average Filter SRAM BW: ' + "{:.3f}".format(avg_filter_sram_bw) + \
                      ' words/cycle')
                if self.conf.sparsity_support is True:
                    print('Average Filter Metadata SRAM BW: ' + \
                          "{:.3f}".format(avg_filter_metadata_sram_bw) + ' words/cycle')
                print('Average OFMAP SRAM BW: ' + "{:.3f}".format(avg_ofmap_sram_bw) + \
                      ' words/cycle')
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_dram_bw) + \
                      ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_dram_bw) + \
                      ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_dram_bw) + \
                      ' words/cycle')

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path)
                if self.verbose:
                    print('Done!')

    #
    def generate_reports(self):
        """
        Method to generate the report files for scalesim run if the runs are already completed. For
        each layer, this method collects the report data from single_layer_sim objects and then
        prints them out into COMPUTE_REPORT.csv, BANDWIDTH_REPORT.csv, DETAILED_ACCESS_REPORT.csv
        and SPARSE_REPORT.csv files.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        # Determine if GPU mode is active
        gpu_mode = self.conf.is_gpu_mode()
        tensor_cores = self.conf.get_tensor_cores() if gpu_mode else 1
        operand_size = self.conf.get_operand_size() if gpu_mode else 1
        clock_freq_mhz = self.conf.get_clock_freq_mhz() if gpu_mode else 1000
        num_sms = self.conf.get_num_sms() if gpu_mode else 1

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = ('LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %,'
                  ' Compute Util %,\n')
        compute_report.write(header)
        
        # Create TIME_REPORT.csv for linear model time conversion
        time_report_name = self.top_path + '/TIME_REPORT.csv'
        time_report = open(time_report_name, 'w')
        time_report.write('LayerID, Time (us),\n')

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        if self.conf.sparsity_support is True:
            header = ('LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg FILTER Metadata SRAM BW,'
                      ' Avg OFMAP SRAM BW, ')
        else:
            header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '
        header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'
        bandwidth_report.write(header)

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, '
        header += 'SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, '
        header += 'SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, '
        header += 'DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, '
        header += 'DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, '
        header += 'DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n'
        detail_report.write(header)

        if self.conf.sparsity_support is True:
            sparse_report_name = self.top_path + '/SPARSE_REPORT.csv'
            sparse_report = open(sparse_report_name, 'w')
            header = 'LayerID, '
            header += 'Sparsity Representation, '
            header += ('Original Filter Storage, New Storage (Filter+Metadata),'
                       ' Filter Metadata Storage, ')
            header += 'Avg FILTER Metadata SRAM BW, '
            header += '\n'
            sparse_report.write(header)

        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            comp_items = list(single_layer_obj.get_compute_report_items())
            bw_items = list(single_layer_obj.get_bandwidth_report_items())
            det_items = list(single_layer_obj.get_detail_report_items())

            if gpu_mode:
                # GPU Scaling: mathematically divide base cycles by TensorCores and calculate memory bounds.
                # comp_items = [Total Cycles, Compute Cycles, Stall Cycles, Util %, Mapping Eff %]
                base_compute_cycles = comp_items[1]
                
                scaled_compute_cycles = base_compute_cycles / tensor_cores
                
                # To find memory bound, sum all SRAM reads/writes from det_items
                # det_items = [
                #   0, 1, 2(IFMAP reads), 3, 4, 5(Filter reads), 6, 7, 8(OFMAP writes)
                #   9, 10, 11(DRAM reads), ...
                # ]
                sram_ifmap_reads = det_items[2]
                sram_filter_reads = det_items[5]
                sram_ofmap_writes = det_items[8]
                
                total_sram_words = sram_ifmap_reads + sram_filter_reads + sram_ofmap_writes
                total_sram_bytes = total_sram_words * operand_size
                
                # Peak bandwidth in bytes per cycle
                max_bw_bytes_per_cycle = (self.conf.get_total_bandwidth_gbs() * 1e9) / (clock_freq_mhz * 1e6)
                memory_bound_cycles = total_sram_bytes / max_bw_bytes_per_cycle if max_bw_bytes_per_cycle > 0 else 0
                
                final_cycles = max(scaled_compute_cycles, memory_bound_cycles)
                final_stall_cycles = max(0, final_cycles - scaled_compute_cycles)
                
                # Update compute items
                comp_items[0] = final_cycles # Total Cycles (incl prefetch)
                comp_items[1] = final_cycles # Total Cycles
                comp_items[2] = final_stall_cycles # Stall Cycles
                
                # Re-calculate Util
                base_util = comp_items[3]
                comp_items[3] = (base_util * base_compute_cycles) / (final_cycles * tensor_cores) if final_cycles > 0 else 0
                
                # Update bandwidth items (bytes/cycle or words/cycle)
                if self.conf.sparsity_support is True:
                    bw_items[0] = sram_ifmap_reads / final_cycles if final_cycles > 0 else 0
                    bw_items[1] = sram_filter_reads / final_cycles if final_cycles > 0 else 0
                    bw_items[3] = sram_ofmap_writes / final_cycles if final_cycles > 0 else 0
                else:
                    bw_items[0] = sram_ifmap_reads / final_cycles if final_cycles > 0 else 0
                    bw_items[1] = sram_filter_reads / final_cycles if final_cycles > 0 else 0
                    bw_items[2] = sram_ofmap_writes / final_cycles if final_cycles > 0 else 0
                
                # Update detail items with final cycles
                det_items[1] = final_cycles # IFMAP stop
                det_items[4] = final_cycles # Filter stop
                det_items[7] = final_cycles # OFMAP stop
                
                # Save scaled final cycle to the single_layer_obj so get_total_cycles can read it
                single_layer_obj.total_cycles_for_gpu = final_cycles

            log = str(lid) +', '
            log += ', '.join([str(x) for x in comp_items])
            log += ',\n'
            compute_report.write(log)
            
            # Generate TIME_REPORT entry using linear model
            total_cycles = comp_items[1]
            time_linear_model = self.conf.get_time_linear_model()
            
            dataflow = self.conf.get_dataflow()
            s_row, s_col, t_time = self.topo.get_spatiotemporal_dims(layer_id=lid, df=dataflow)
            
            if time_linear_model == 'TPUv4':
                time_us = tpuv4_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv5e':
                time_us = tpuv5e_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv6e':
                time_us = tpuv6e_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'GA10b':
                time_us = ga10b_linear_model(total_cycles, s_row, s_col, t_time)
            else:
                time_us = total_cycles
            
            time_log = str(lid) + ', ' + str(time_us) + ',\n'
            time_report.write(time_log)

            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bw_items])
            log += ',\n'
            bandwidth_report.write(log)

            log = str(lid) + ', '
            log += ', '.join([str(x) for x in det_items])
            log += ',\n'
            detail_report.write(log)

            if self.conf.sparsity_support is True:
                sparse_report_items_this_layer = single_layer_obj.get_sparse_report_items()
                log = str(lid) + ', ' + self.conf.sparsity_representation + ', '
                log += ', '.join([str(x) for x in sparse_report_items_this_layer])
                log += ',\n'
                sparse_report.write(log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()
        time_report.close()
        if self.conf.sparsity_support is True:
            sparse_report.close()

    #
    def get_total_cycles(self):
        """
        Method which aggregates the total cycles (both compute and stall) across all the layers for
        the given workload.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            if hasattr(layer_obj, 'total_cycles_for_gpu'):
                cycles_this_layer = int(layer_obj.total_cycles_for_gpu)
            else:
                cycles_this_layer = int(layer_obj.get_compute_report_items()[0])
            total_cycles += cycles_this_layer

        return total_cycles
