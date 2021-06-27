# Copyright (c) 2006-2007 The Regents of The University of Michigan
# Copyright (c) 2009 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Brad Beckmann

import math
import m5
from m5.objects import *
from m5.defines import buildEnv
from Ruby import create_topology
from Ruby import send_evicts

#
# Note: the L1 Cache latency is only used by the sequencer on fast path hits
#
class L1Cache(RubyCache):
    latency = 2

#
# Note: the L2 Cache latency is not currently used
#
class L2Cache(RubyCache):
    latency = 10

def define_options(parser):
    parser.add_option("--l1-retries", type="int", default=1,
                      help="Token_CMP: # of l1 retries before going persistent")
    parser.add_option("--timeout-latency", type="int", default=300,
                      help="Token_CMP: cycles until issuing again");
    parser.add_option("--disable-dyn-timeouts", action="store_true",
          help="Token_CMP: disable dyanimc timeouts, use fixed latency instead")
    parser.add_option("--allow-atomic-migration", action="store_true",
          help="allow migratory sharing for atomic only accessed blocks")
    
def create_system(options, full_system, system, dma_ports, ruby_system):
    
    if buildEnv['PROTOCOL'] != 'MOESI_CMP_token':
        panic("This script requires the MOESI_CMP_token protocol to be built.")

    #
    # number of tokens that the owner passes to requests so that shared blocks can
    # respond to read requests
    #
    n_tokens = options.num_cpus + 1

    cpu_sequencers = []
    
    #
    # The ruby network creation expects the list of nodes in the system to be
    # consistent with the NetDest list.  Therefore the l1 controller nodes must be
    # listed before the directory nodes and directory nodes before dma nodes, etc.
    #
    l1_cntrl_nodes = []
    l2_cntrl_nodes = []
    dir_cntrl_nodes = []
    dma_cntrl_nodes = []

    #
    # Must create the individual controllers before the network to ensure the
    # controller constructors are called before the network constructor
    #
    l2_bits = int(math.log(options.num_l2caches, 2))
    block_size_bits = int(math.log(options.cacheline_size, 2))
    
    for i in xrange(options.num_cpus):
        #
        # First create the Ruby objects associated with this cpu
        #
        l1i_cache = L1Cache(size = options.l1i_size,
                            assoc = options.l1i_assoc,
                            start_index_bit = block_size_bits)
        l1d_cache = L1Cache(size = options.l1d_size,
                            assoc = options.l1d_assoc,
                            start_index_bit = block_size_bits)

        l1_cntrl = L1Cache_Controller(version = i,
                                      L1Icache = l1i_cache,
                                      L1Dcache = l1d_cache,
                                      l2_select_num_bits = l2_bits,
                                      N_tokens = n_tokens,
                                      retry_threshold = \
                                        options.l1_retries,
                                      fixed_timeout_latency = \
                                        options.timeout_latency,
                                      dynamic_timeout_enabled = \
                                        not options.disable_dyn_timeouts,
                                      no_mig_atomic = not \
                                        options.allow_atomic_migration,
                                      send_evictions = send_evicts(options),
                                      transitions_per_cycle = options.ports,
                                      clk_domain=system.cpu[i].clk_domain,
                                      ruby_system = ruby_system)

        cpu_seq = RubySequencer(version = i,
                                icache = l1i_cache,
                                dcache = l1d_cache,
                                clk_domain=system.cpu[i].clk_domain,
                                ruby_system = ruby_system)

        l1_cntrl.sequencer = cpu_seq
        exec("ruby_system.l1_cntrl%d = l1_cntrl" % i)

        # Add controllers and sequencers to the appropriate lists
        cpu_sequencers.append(cpu_seq)
        l1_cntrl_nodes.append(l1_cntrl)

        # Connect the L1 controllers and the network
        l1_cntrl.requestFromL1Cache =  ruby_system.network.slave
        l1_cntrl.responseFromL1Cache =  ruby_system.network.slave
        l1_cntrl.persistentFromL1Cache =  ruby_system.network.slave

        l1_cntrl.requestToL1Cache =  ruby_system.network.master
        l1_cntrl.responseToL1Cache =  ruby_system.network.master
        l1_cntrl.persistentToL1Cache =  ruby_system.network.master


    l2_index_start = block_size_bits + l2_bits

    for i in xrange(options.num_l2caches):
        #
        # First create the Ruby objects associated with this cpu
        #
        l2_cache = L2Cache(size = options.l2_size,
                           assoc = options.l2_assoc,
                           start_index_bit = l2_index_start)

        l2_cntrl = L2Cache_Controller(version = i,
                                      L2cache = l2_cache,
                                      N_tokens = n_tokens,
                                      transitions_per_cycle = options.ports,
                                      ruby_system = ruby_system)
        
        exec("ruby_system.l2_cntrl%d = l2_cntrl" % i)
        l2_cntrl_nodes.append(l2_cntrl)

        # Connect the L2 controllers and the network
        l2_cntrl.GlobalRequestFromL2Cache = ruby_system.network.slave
        l2_cntrl.L1RequestFromL2Cache = ruby_system.network.slave
        l2_cntrl.responseFromL2Cache = ruby_system.network.slave

        l2_cntrl.GlobalRequestToL2Cache = ruby_system.network.master
        l2_cntrl.L1RequestToL2Cache = ruby_system.network.master
        l2_cntrl.responseToL2Cache = ruby_system.network.master
        l2_cntrl.persistentToL2Cache = ruby_system.network.master


    phys_mem_size = sum(map(lambda r: r.size(), system.mem_ranges))
    assert(phys_mem_size % options.num_dirs == 0)
    mem_module_size = phys_mem_size / options.num_dirs

    # Run each of the ruby memory controllers at a ratio of the frequency of
    # the ruby system
    # clk_divider value is a fix to pass regression.
    ruby_system.memctrl_clk_domain = DerivedClockDomain(
                                          clk_domain=ruby_system.clk_domain,
                                          clk_divider=3)

    for i in xrange(options.num_dirs):
        dir_size = MemorySize('0B')
        dir_size.value = mem_module_size

        dir_cntrl = Directory_Controller(version = i,
                                         directory = RubyDirectoryMemory(
                                             version = i, size = dir_size),
                                         l2_select_num_bits = l2_bits,
                                         transitions_per_cycle = options.ports,
                                         ruby_system = ruby_system)

        exec("ruby_system.dir_cntrl%d = dir_cntrl" % i)
        dir_cntrl_nodes.append(dir_cntrl)

        # Connect the directory controllers and the network
        dir_cntrl.requestToDir = ruby_system.network.master
        dir_cntrl.responseToDir = ruby_system.network.master
        dir_cntrl.persistentToDir = ruby_system.network.master
        dir_cntrl.dmaRequestToDir = ruby_system.network.master

        dir_cntrl.requestFromDir = ruby_system.network.slave
        dir_cntrl.responseFromDir = ruby_system.network.slave
        dir_cntrl.persistentFromDir = ruby_system.network.slave
        dir_cntrl.dmaResponseFromDir = ruby_system.network.slave


    for i, dma_port in enumerate(dma_ports):
        #
        # Create the Ruby objects associated with the dma controller
        #
        dma_seq = DMASequencer(version = i,
                               ruby_system = ruby_system,
                               slave = dma_port)
        
        dma_cntrl = DMA_Controller(version = i,
                                   dma_sequencer = dma_seq,
                                   transitions_per_cycle = options.ports,
                                   ruby_system = ruby_system)

        exec("ruby_system.dma_cntrl%d = dma_cntrl" % i)
        dma_cntrl_nodes.append(dma_cntrl)

        # Connect the dma controller to the network
        dma_cntrl.responseFromDir = ruby_system.network.master
        dma_cntrl.reqToDirectory = ruby_system.network.slave

    all_cntrls = l1_cntrl_nodes + \
                 l2_cntrl_nodes + \
                 dir_cntrl_nodes + \
                 dma_cntrl_nodes

    # Create the io controller and the sequencer
    if full_system:
        io_seq = DMASequencer(version=len(dma_ports), ruby_system=ruby_system)
        ruby_system._io_port = io_seq
        io_controller = DMA_Controller(version = len(dma_ports),
                                       dma_sequencer = io_seq,
                                       ruby_system = ruby_system)
        ruby_system.io_controller = io_controller

        # Connect the dma controller to the network
        io_controller.responseFromDir = ruby_system.network.master
        io_controller.reqToDirectory = ruby_system.network.slave

        all_cntrls = all_cntrls + [io_controller]


    topology = create_topology(all_cntrls, options)
    return (cpu_sequencers, dir_cntrl_nodes, topology)