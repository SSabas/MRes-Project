\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.24273 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.936263 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.936895 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.985555 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.14838 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03074 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.82462 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.672255 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.35979 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20996 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.44428 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.924366 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.17768 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.922003 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.685883 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.48814 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.05816 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.980189 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.778794 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.718119 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.908566 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.765729 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.906615 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.899634 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.99943 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.01631 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.04847 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.23153 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.820877 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.62556 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.1856 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.23933 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15324 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.12368 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.00464 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.636561 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.88103 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.87685 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.30445 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.943492 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.994503 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.09483 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.973932 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.06618 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.614946 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.595591 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.43423 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.786391 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.20815 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.763282 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.39583 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.798439 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.559938 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.77932 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.11298 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.36602 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.958386 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.620518 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.90379 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.719557 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.46473 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.879399 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.1524 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.10471 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.52822 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.67833 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.815312 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.551759 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.61392 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.27155 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.27081 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.158 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.847998 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.45366 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.907343 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.736748 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.40677 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.914111 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.95065 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.06892 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.994688 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.03564 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.558203 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.748338 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.13232 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.23598 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.4423 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.706264 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.45999 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.745316 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.44534 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.91129 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.26193 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.58437 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.02152 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.73841 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.889924 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.599244 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.06862 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.807816 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.22848 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.20615 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.54703 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.75548 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.766661 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.484608 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.83265 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.14476 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.57482 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.20821 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.861893 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.460004 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.890718 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.787828 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.21078 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.933297 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.08636 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.966291 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.02063 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.937528 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.658922 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.750575 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.16695 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.997188 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.33174 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.858455 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.12843 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.781357 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.626598 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.6781 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.23624 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.28615 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.973119 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.733817 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.984453 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.874894 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.08839 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.972588 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.11073 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.26591 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.2628 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.53686 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.856361 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.673896 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.50864 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.10755 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.43812 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.17986 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.928065 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.604737 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.87681 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.89872 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17224 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.978212 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.802542 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.27885 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.851142 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.33547 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.898696 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.606362 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.13785 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.13372 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.925126 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.688837 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.58347 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.21263 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.715241 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.19144 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.921896 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.57373 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.38418 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.957945 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.836529 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.675288 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.2599 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.634576 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.14442 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.71402 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.68989 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.19069 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.974081 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.610799 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.67671 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.54597 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.967391 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.11344 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.910716 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.613728 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.22556 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.668861 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18026 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.933603 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.973941 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.965999 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05367 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06848 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.728051 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.789021 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.145 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.998641 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.14297 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.851724 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08126 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.80686 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.771557 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.38409 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.08236 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.41995 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.10131 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.775779 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.01375 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.887283 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07942 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.876487 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.07754 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.11242 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.25106 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.40347 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.861417 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.601684 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.37792 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.968314 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.16585 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.03171 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.921619 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.667095 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.848759 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.876887 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20653 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.947888 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.88674 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.22126 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.97765 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.08508 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.751532 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.807957 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.21029 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.23031 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.14735 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.829672 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.38569 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.02075 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.626375 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.39604 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.01434 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.21823 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.15688 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.825628 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.883466 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.636165 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.06319 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.740591 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.09489 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.938119 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.47525 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.33825 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.955914 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.593613 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.78952 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.30948 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.33433 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.1727 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.956255 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.620302 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.08363 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.799961 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2252 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.966273 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.782408 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.27411 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.887448 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.221 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.888123 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.740525 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.66802 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.1369 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.986294 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.758406 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.53626 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.19965 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.728727 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.32433 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.966785 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.89558 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.26467 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.870149 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.04107 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.613876 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.43277 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.631156 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.12069 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.841844 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.75521 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.34395 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.93087 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.580598 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.85858 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.51583 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.27107 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.27008 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.03959 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.635137 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.48064 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.680895 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21146 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.943793 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.963454 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.00515 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02011 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.952153 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.777298 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.915745 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.04023 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.12784 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.30915 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.893697 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13309 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.881893 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.642562 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.4178 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.11731 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.23218 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.00541 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.823582 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.95482 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.858809 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.979247 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.997363 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.17551 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.13255 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.23249 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.28424 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.830588 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.64002 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.38179 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.02976 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.42326 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.17547 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.933639 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.724789 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.910597 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.91683 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15454 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.941987 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.985238 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.01969 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.01475 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.03675 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.810082 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.832649 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.15113 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02157 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12951 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.837475 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.09436 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.88653 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.757821 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.37186 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00816 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.19327 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.06491 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.873948 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.965088 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.822359 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.07284 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.934599 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06335 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.07233 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.26522 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.26521 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.930613 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.685264 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.31374 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.16277 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.13525 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.12531 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.950147 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.719167 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01059 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.895671 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.238999492613308 z_k1_s0000
                                 - 0.37599898294498 z_k1_s0001
                                 - 0.168999160619625 z_k1_s0002
                                 - 0.173999115155214 z_k1_s0003
                                 - 0.336997671755236 z_k1_s0004
                                 - 0.22799932145629 z_k1_s0005
                                 - 0.21599966300978 z_k1_s0006
                                 - 0.197999206844581 z_k1_s0007
                                 - 0.072999923858768 z_k1_s0010
                                 - 0.0709999259448292 z_k1_s0011
                                 - 0.199999791393885 z_k1_s0012
                                 - 0.111999883180576 z_k1_s0013
                                 - 0.165000431399627 z_k1_s0014
                                 - 0.231000362559609 z_k1_s0015
                                 - 0.137999856061781 z_k1_s0016
                                 - 0.217000377162037 z_k1_s0017
                                 - 0.178000200752039 z_k1_s0020
                                 - 0.110000043999591 z_k1_s0021
                                 - 0.162999687806744 z_k1_s0022
                                 - 0.112999991688682 z_k1_s0023
                                 - 0.325999375613487 z_k1_s0024
                                 - 0.269999939017212 z_k1_s0025
                                 - 0.111999957492329 z_k1_s0026
                                 - 0.276999868591747 z_k1_s0027
                                 - 0.231000177499898 z_k1_s0030
                                 - 0.222000326129623 z_k1_s0031
                                 - 0.184999986608292 z_k1_s0032
                                 - 0.116999949911771 z_k1_s0033
                                 - 0.125999801282046 z_k1_s0034
                                 - 0.171999821073815 z_k1_s0035
                                 - 0.367999816134484 z_k1_s0036
                                 - 0.29000053392598 z_k1_s0037
                                 - 0.342000633649008 z_k1_s0040
                                 - 0.748001330594817 z_k1_s0041
                                 - 0.398999163754657 z_k1_s0042
                                 - 0.192000189890406 z_k1_s0043
                                 - 0.480002050228533 z_k1_s0044
                                 - 0.515000263169185 z_k1_s0045
                                 - 0.180000217409818 z_k1_s0046
                                 - 0.295000557624746 z_k1_s0047
                                 - 0.4000009609019 z_k1_s0050
                                 - 0.471000381250568 z_k1_s0051
                                 - 0.486001667636755 z_k1_s0052
                                 - 0.607000472596769 z_k1_s0053
                                 - 0.167000298206349 z_k1_s0054
                                 - 0.176000187436391 z_k1_s0055
                                 - 0.315000536059829 z_k1_s0056
                                 - 0.320001945523747 z_k1_s0057
                                 - 0.472002117252759 z_k1_s0060
                                 - 0.308001033545528 z_k1_s0061
                                 - 0.274999847163291 z_k1_s0062
                                 - 0.376002150693894 z_k1_s0063
                                 - 0.286001155292705 z_k1_s0064
                                 - 0.516001873758405 z_k1_s0065
                                 - 0.381001749659625 z_k1_s0066
                                 - 0.124000184972368 z_k1_s0067
                                 - 0.0959999111044058 z_k1_s0070
                                 - 0.0859999719643467 z_k1_s0071
                                 - 0.0779999105723354 z_k1_s0072
                                 - 0.154000493796444 z_k1_s0073
                                 - 0.237999762413012 z_k1_s0074
                                 - 0.158999500166787 z_k1_s0075
                                 - 0.268999656307168 z_k1_s0076
                                 - 0.295999519505319 z_k1_s0077
                                 - 0.15500050643067 z_k1_s0100
                                 - 0.175999607688823 z_k1_s0101
                                 - 0.181000107702813 z_k1_s0102
                                 - 0.0870000251416783 z_k1_s0103
                                 - 0.105000004491676 z_k1_s0104
                                 - 0.0680000528889047 z_k1_s0105
                                 - 0.0580000167611189 z_k1_s0106
                                 - 0.240999681869398 z_k1_s0107
                                 - 0.120000383157675 z_k1_s0110
                                 - 0.139000134449285 z_k1_s0111
                                 - 0.0470000056950794 z_k1_s0112
                                 - 0.0980001066620869 z_k1_s0113
                                 - 0.0360000324472967 z_k1_s0114
                                 - 0.0569999963748828 z_k1_s0115
                                 - 0.171000236624665 z_k1_s0116
                                 - 0.156999985672922 z_k1_s0117
                                 - 0.109000046274704 z_k1_s0120
                                 - 0.0829999998752118 z_k1_s0121
                                 - 0.409999788299109 z_k1_s0122
                                 - 0.274000704610169 z_k1_s0123
                                 - 0.121000027259061 z_k1_s0124
                                 - 0.239999619687126 z_k1_s0125
                                 - 0.233000375380043 z_k1_s0126
                                 - 0.282000341532858 z_k1_s0127
                                 - 0.212000202633915 z_k1_s0130
                                 - 0.287000540288247 z_k1_s0131
                                 - 0.100000235405111 z_k1_s0132
                                 - 0.39000175580253 z_k1_s0133
                                 - 0.165000742839532 z_k1_s0134
                                 - 0.152000415811948 z_k1_s0135
                                 - 0.185001369882353 z_k1_s0136
                                 - 0.120000379146432 z_k1_s0137
                                 - 0.256000283452696 z_k1_s0140
                                 - 0.336000711838209 z_k1_s0141
                                 - 0.304999998912098 z_k1_s0142
                                 - 0.469000346356938 z_k1_s0143
                                 - 0.244000258029903 z_k1_s0144
                                 - 0.231000101038431 z_k1_s0145
                                 - 0.1320000207505 z_k1_s0146
                                 - 0.616001305036717 z_k1_s0147
                                 - 0.169999741789737 z_k1_s0150
                                 - 0.149000643404252 z_k1_s0151
                                 - 0.0690000884676671 z_k1_s0152
                                 - 0.234999884650604 z_k1_s0153
                                 - 0.286000628587814 z_k1_s0154
                                 - 0.0630000331574184 z_k1_s0155
                                 - 0.202000128044429 z_k1_s0156
                                 - 0.1950004285826 z_k1_s0157
                                 - 0.390001522954819 z_k1_s0160
                                 - 0.19200024319279 z_k1_s0161
                                 - 0.181000191350468 z_k1_s0162
                                 - 0.738002028729295 z_k1_s0163
                                 - 0.268000317794087 z_k1_s0164
                                 - 0.743000004201655 z_k1_s0165
                                 - 0.543999475945394 z_k1_s0166
                                 - 0.410000356849073 z_k1_s0167
                                 - 0.122999994116067 z_k1_s0170
                                 - 0.40099955147786 z_k1_s0171
                                 - 0.157000611565641 z_k1_s0172
                                 - 0.0829999717043537 z_k1_s0173
                                 - 0.206999891580762 z_k1_s0174
                                 - 0.117999916514651 z_k1_s0175
                                 - 0.224999752066129 z_k1_s0176
                                 - 0.182000251573201 z_k1_s0177
                                 - 0.832000692191112 z_k1_s1000
                                 - 0.516000602599469 z_k1_s1001
                                 - 0.169000103051282 z_k1_s1002
                                 - 0.192000252167272 z_k1_s1003
                                 - 0.224000093928283 z_k1_s1004
                                 - 0.430999333170086 z_k1_s1005
                                 - 0.275000254785309 z_k1_s1006
                                 - 0.365001386839737 z_k1_s1007
                                 - 0.706999388768925 z_k1_s1010
                                 - 0.352999928099848 z_k1_s1011
                                 - 0.31499987127325 z_k1_s1012
                                 - 0.547998289478887 z_k1_s1013
                                 - 0.811001332757095 z_k1_s1014
                                 - 0.687000449386254 z_k1_s1015
                                 - 0.470998643256462 z_k1_s1016
                                 - 0.2519998970186 z_k1_s1017
                                 - 0.355998981101144 z_k1_s1020
                                 - 1.29199602142616 z_k1_s1021
                                 - 0.738000830048846 z_k1_s1022
                                 - 0.292999240379128 z_k1_s1023
                                 - 0.5379971908783 z_k1_s1024
                                 - 0.553000847708159 z_k1_s1025
                                 - 0.950998614759755 z_k1_s1026
                                 - 0.48499867331237 z_k1_s1027
                                 - 0.195000008707205 z_k1_s1030
                                 - 0.38300044801192 z_k1_s1031
                                 - 0.96899982990487 z_k1_s1032
                                 - 0.795001792607654 z_k1_s1033
                                 - 0.529001839300435 z_k1_s1034
                                 - 0.350000088841194 z_k1_s1035
                                 - 0.587001185066173 z_k1_s1036
                                 - 0.271000403266725 z_k1_s1037
                                 - 0.42100185188443 z_k1_s1040
                                 - 0.626999548127382 z_k1_s1041
                                 - 0.626999548127382 z_k1_s1042
                                 - 0.249999824304378 z_k1_s1043
                                 - 0.70000159325212 z_k1_s1044
                                 - 0.703999562985125 z_k1_s1045
                                 - 0.318999889696379 z_k1_s1046
                                 - 0.361999918824728 z_k1_s1047
                                 - 0.835996533455573 z_k1_s1050
                                 - 1.24299605684438 z_k1_s1051
                                 - 0.294999181908451 z_k1_s1052
                                 - 0.288999114307882 z_k1_s1053
                                 - 0.344998824849594 z_k1_s1054
                                 - 0.376998725187497 z_k1_s1055
                                 - 0.36199901638454 z_k1_s1056
                                 - 0.85499812811944 z_k1_s1057
                                 - 1.58200651987978 z_k1_s1060
                                 - 3.13400010968962 z_k1_s1061
                                 - 1.17700079538769 z_k1_s1062
                                 - 2.23899807044875 z_k1_s1063
                                 - 2.26700176670649 z_k1_s1064
                                 - 1.45599686322625 z_k1_s1065
                                 - 0.863000804877578 z_k1_s1066
                                 - 1.23500067810028 z_k1_s1067
                                 - 0.580997635042966 z_k1_s1070
                                 - 0.956997392260853 z_k1_s1071
                                 - 0.48300138674404 z_k1_s1072
                                 - 0.520998256167333 z_k1_s1073
                                 - 0.27299991108207 z_k1_s1074
                                 - 0.886998116905948 z_k1_s1075
                                 - 0.486997695738494 z_k1_s1076
                                 - 0.372999788274105 z_k1_s1077
                                 - 0.330999642564201 z_k1_s1100
                                 - 0.545001926191334 z_k1_s1101
                                 - 0.313000247728169 z_k1_s1102
                                 - 0.189000366793209 z_k1_s1103
                                 - 0.404000204128116 z_k1_s1104
                                 - 0.128000177288365 z_k1_s1105
                                 - 0.324999844285524 z_k1_s1106
                                 - 0.35000201962002 z_k1_s1107
                                 - 0.367000344404344 z_k1_s1110
                                 - 0.235000356021858 z_k1_s1111
                                 - 0.220999784663095 z_k1_s1112
                                 - 0.201000389150613 z_k1_s1113
                                 - 0.290000185431223 z_k1_s1114
                                 - 0.173999912358729 z_k1_s1115
                                 - 0.257000685585614 z_k1_s1116
                                 - 0.244000581252488 z_k1_s1117
                                 - 0.417000894614752 z_k1_s1120
                                 - 0.239000038797813 z_k1_s1121
                                 - 0.342000293245331 z_k1_s1122
                                 - 0.264998859506316 z_k1_s1123
                                 - 0.321999316547076 z_k1_s1124
                                 - 0.273998931670647 z_k1_s1125
                                 - 0.336000245135777 z_k1_s1126
                                 - 0.25400040397162 z_k1_s1127
                                 - 0.399000091840657 z_k1_s1130
                                 - 0.451999745017363 z_k1_s1131
                                 - 0.38900123841146 z_k1_s1132
                                 - 0.370000551895005 z_k1_s1133
                                 - 0.286000155586316 z_k1_s1134
                                 - 0.238000215624307 z_k1_s1135
                                 - 0.277000041499646 z_k1_s1136
                                 - 0.453999515703202 z_k1_s1137
                                 - 1.25499873994183 z_k1_s1140
                                 - 1.02100177064006 z_k1_s1141
                                 - 1.25400254060469 z_k1_s1142
                                 - 1.33700206037799 z_k1_s1143
                                 - 0.303999750222924 z_k1_s1144
                                 - 0.704999283403583 z_k1_s1145
                                 - 0.940998173873304 z_k1_s1146
                                 - 0.507999399484482 z_k1_s1147
                                 - 0.177000108045009 z_k1_s1150
                                 - 0.20399920064841 z_k1_s1151
                                 - 0.111999961940336 z_k1_s1152
                                 - 0.418000366113081 z_k1_s1153
                                 - 0.303000816727948 z_k1_s1154
                                 - 0.264000969545164 z_k1_s1155
                                 - 0.238999063504754 z_k1_s1156
                                 - 0.287000879422191 z_k1_s1157
                                 - 0.172000447998119 z_k1_s1160
                                 - 0.195000576028288 z_k1_s1161
                                 - 0.377001948063654 z_k1_s1162
                                 - 0.797003475525372 z_k1_s1163
                                 - 0.247000587608774 z_k1_s1164
                                 - 0.177000510565643 z_k1_s1165
                                 - 0.242000791341983 z_k1_s1166
                                 - 0.456000060582608 z_k1_s1167
                                 - 0.43900252447127 z_k1_s1170
                                 - 0.370000794872204 z_k1_s1171
                                 - 0.359001038198297 z_k1_s1172
                                 - 0.296000786698143 z_k1_s1173
                                 - 0.73400199665024 z_k1_s1174
                                 - 0.578002774177191 z_k1_s1175
                                 - 0.575001846627261 z_k1_s1176
                                 - 0.419002624154212 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.939669 w_k1_ADM_s000
                                 + 1.11904 w_k1_AXP_s000
                                 + 1.15527 w_k1_DIS_s000
                                 + 1.17114 w_k1_FDX_s000
                                 + 0.887732 w_k1_IBM_s000
                                 + 1.28594 w_k1_INTC_s000
                                 + 1.06163 w_k1_KO_s000
                                 + 0.691532 w_k1_MAT_s000
                                 + 1.08158 w_k1_MSFT_s000
                                 + 1.05403 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.788086 w_k1_ADM_s000
                                 + 0.958023 w_k1_AXP_s000
                                 + 0.847818 w_k1_DIS_s000
                                 + 0.988961 w_k1_FDX_s000
                                 + 0.775426 w_k1_IBM_s000
                                 + 1.11999 w_k1_INTC_s000
                                 + 0.826459 w_k1_KO_s000
                                 + 0.708257 w_k1_MAT_s000
                                 + 0.873315 w_k1_MSFT_s000
                                 + 0.900068 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.01391 w_k1_ADM_s000
                                 + 0.955432 w_k1_AXP_s000 + 1.02 w_k1_DIS_s000
                                 + 1.02747 w_k1_FDX_s000
                                 + 1.31281 w_k1_IBM_s000
                                 + 1.08833 w_k1_INTC_s000
                                 + 1.08168 w_k1_KO_s000 + 1.30453 w_k1_MAT_s000
                                 + 0.952294 w_k1_MSFT_s000
                                 + 0.992392 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.945777 w_k1_ADM_s000
                                 + 0.982938 w_k1_AXP_s000
                                 + 0.83558 w_k1_DIS_s000
                                 + 0.820483 w_k1_FDX_s000
                                 + 1.17384 w_k1_IBM_s000
                                 + 1.08696 w_k1_INTC_s000 + 1.0877 w_k1_KO_s000
                                 + 0.954001 w_k1_MAT_s000
                                 + 0.878058 w_k1_MSFT_s000
                                 + 0.950049 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.615639 w_k1_ADM_s000
                                 + 0.608568 w_k1_AXP_s000
                                 + 0.653184 w_k1_DIS_s000
                                 + 0.730546 w_k1_FDX_s000
                                 + 0.7287 w_k1_IBM_s000
                                 + 0.866351 w_k1_INTC_s000
                                 + 0.792042 w_k1_KO_s000
                                 + 0.795948 w_k1_MAT_s000
                                 + 0.763731 w_k1_MSFT_s000
                                 + 0.820872 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.0284 w_k1_ADM_s000
                                 + 1.28724 w_k1_AXP_s000
                                 + 1.57879 w_k1_DIS_s000
                                 + 1.24144 w_k1_FDX_s000
                                 + 1.47721 w_k1_IBM_s000
                                 + 1.26807 w_k1_INTC_s000
                                 + 1.74303 w_k1_KO_s000 + 1.47827 w_k1_MAT_s000
                                 + 1.27257 w_k1_MSFT_s000
                                 + 1.22393 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.781425 w_k1_ADM_s000
                                 + 1.2339 w_k1_AXP_s000 + 1.27052 w_k1_DIS_s000
                                 + 1.09183 w_k1_FDX_s000
                                 + 1.09549 w_k1_IBM_s000
                                 + 1.08851 w_k1_INTC_s000
                                 + 1.07207 w_k1_KO_s000 + 1.37668 w_k1_MAT_s000
                                 + 0.897647 w_k1_MSFT_s000
                                 + 1.04604 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.19422 w_k1_ADM_s000
                                 + 1.3 w_k1_AXP_s000 + 1.83867 w_k1_DIS_s000
                                 + 1.41316 w_k1_FDX_s000
                                 + 0.983904 w_k1_IBM_s000
                                 + 1.38801 w_k1_INTC_s000
                                 + 1.26554 w_k1_KO_s000
                                 + 0.735166 w_k1_MAT_s000
                                 + 1.24412 w_k1_MSFT_s000 + 1.2199 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.10995 w_k1_ADM_s001
                                 + 0.861597 w_k1_AXP_s001
                                 + 1.09357 w_k1_DIS_s001
                                 + 0.887278 w_k1_FDX_s001
                                 + 1.34827 w_k1_IBM_s001
                                 + 0.915021 w_k1_INTC_s001
                                 + 1.04422 w_k1_KO_s001
                                 + 0.859512 w_k1_MAT_s001
                                 + 1.021 w_k1_MSFT_s001 + 0.993516 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.04628 w_k1_ADM_s001
                                 + 1.04111 w_k1_AXP_s001 + 1.2187 w_k1_DIS_s001
                                 + 1.08546 w_k1_FDX_s001
                                 + 0.914462 w_k1_IBM_s001
                                 + 1.19398 w_k1_INTC_s001
                                 + 1.00301 w_k1_KO_s001
                                 + 0.802143 w_k1_MAT_s001
                                 + 1.0778 w_k1_MSFT_s001 + 1.04738 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.36056 w_k1_ADM_s001
                                 + 1.18471 w_k1_AXP_s001
                                 + 1.02113 w_k1_DIS_s001 + 1.0054 w_k1_FDX_s001
                                 + 1.40723 w_k1_IBM_s001
                                 + 1.19613 w_k1_INTC_s001
                                 + 1.03188 w_k1_KO_s001
                                 + 0.975127 w_k1_MAT_s001
                                 + 0.948629 w_k1_MSFT_s001
                                 + 1.08044 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.05317 w_k1_ADM_s001
                                 + 0.909123 w_k1_AXP_s001
                                 + 1.08673 w_k1_DIS_s001
                                 + 0.998235 w_k1_FDX_s001
                                 + 1.07295 w_k1_IBM_s001
                                 + 1.10001 w_k1_INTC_s001
                                 + 0.942011 w_k1_KO_s001
                                 + 0.917396 w_k1_MAT_s001
                                 + 0.940986 w_k1_MSFT_s001
                                 + 0.942681 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.48129 w_k1_ADM_s001
                                 + 1.40634 w_k1_AXP_s001
                                 + 1.80298 w_k1_DIS_s001
                                 + 1.35931 w_k1_FDX_s001 + 1.5359 w_k1_IBM_s001
                                 + 1.34711 w_k1_INTC_s001
                                 + 1.35225 w_k1_KO_s001 + 1.36069 w_k1_MAT_s001
                                 + 1.38093 w_k1_MSFT_s001
                                 + 1.20954 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.888692 w_k1_ADM_s001
                                 + 0.695319 w_k1_AXP_s001
                                 + 0.580414 w_k1_DIS_s001
                                 + 0.691209 w_k1_FDX_s001
                                 + 0.858341 w_k1_IBM_s001
                                 + 0.744793 w_k1_INTC_s001
                                 + 0.794057 w_k1_KO_s001
                                 + 0.596103 w_k1_MAT_s001
                                 + 0.698933 w_k1_MSFT_s001
                                 + 0.784101 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.32481 w_k1_ADM_s001
                                 + 1.21649 w_k1_AXP_s001
                                 + 1.27169 w_k1_DIS_s001
                                 + 1.32225 w_k1_FDX_s001
                                 + 0.984848 w_k1_IBM_s001
                                 + 1.33025 w_k1_INTC_s001
                                 + 1.16533 w_k1_KO_s001
                                 + 0.815351 w_k1_MAT_s001
                                 + 1.23236 w_k1_MSFT_s001
                                 + 1.04088 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.30459 w_k1_ADM_s001
                                 + 0.85812 w_k1_AXP_s001
                                 + 0.794837 w_k1_DIS_s001
                                 + 0.898165 w_k1_FDX_s001
                                 + 0.774633 w_k1_IBM_s001
                                 + 0.908845 w_k1_INTC_s001
                                 + 0.729287 w_k1_KO_s001
                                 + 0.475536 w_k1_MAT_s001
                                 + 0.852537 w_k1_MSFT_s001
                                 + 0.841616 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.974918 w_k1_ADM_s002
                                 + 1.77729 w_k1_AXP_s002
                                 + 2.13501 w_k1_DIS_s002
                                 + 1.51792 w_k1_FDX_s002
                                 + 1.18976 w_k1_IBM_s002
                                 + 1.03576 w_k1_INTC_s002
                                 + 1.02481 w_k1_KO_s002 + 1.22914 w_k1_MAT_s002
                                 + 1.28779 w_k1_MSFT_s002
                                 + 1.10978 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.32332 w_k1_ADM_s002
                                 + 1.49679 w_k1_AXP_s002
                                 + 1.41946 w_k1_DIS_s002
                                 + 1.17208 w_k1_FDX_s002
                                 + 1.45746 w_k1_IBM_s002
                                 + 1.27132 w_k1_INTC_s002
                                 + 1.26591 w_k1_KO_s002 + 1.55517 w_k1_MAT_s002
                                 + 1.1956 w_k1_MSFT_s002 + 1.17949 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.631296 w_k1_ADM_s002
                                 + 0.715958 w_k1_AXP_s002
                                 + 0.749703 w_k1_DIS_s002
                                 + 0.775551 w_k1_FDX_s002
                                 + 0.904567 w_k1_IBM_s002
                                 + 0.770467 w_k1_INTC_s002
                                 + 0.797459 w_k1_KO_s002
                                 + 0.784539 w_k1_MAT_s002
                                 + 0.955079 w_k1_MSFT_s002
                                 + 0.816223 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.868665 w_k1_ADM_s002
                                 + 0.663664 w_k1_AXP_s002
                                 + 0.616019 w_k1_DIS_s002
                                 + 0.757727 w_k1_FDX_s002
                                 + 0.618536 w_k1_IBM_s002
                                 + 0.650428 w_k1_INTC_s002
                                 + 0.618989 w_k1_KO_s002
                                 + 0.423433 w_k1_MAT_s002
                                 + 0.815491 w_k1_MSFT_s002
                                 + 0.70162 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.860762 w_k1_ADM_s002
                                 + 1.11592 w_k1_AXP_s002
                                 + 1.03042 w_k1_DIS_s002
                                 + 0.962471 w_k1_FDX_s002
                                 + 1.09559 w_k1_IBM_s002
                                 + 0.841727 w_k1_INTC_s002
                                 + 0.994128 w_k1_KO_s002
                                 + 1.29637 w_k1_MAT_s002
                                 + 1.00158 w_k1_MSFT_s002 + 1.0123 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.94971 w_k1_ADM_s002
                                 + 1.14367 w_k1_AXP_s002 + 1.3044 w_k1_DIS_s002
                                 + 1.18761 w_k1_FDX_s002
                                 + 0.913642 w_k1_IBM_s002
                                 + 1.02621 w_k1_INTC_s002
                                 + 1.02892 w_k1_KO_s002
                                 + 0.666603 w_k1_MAT_s002
                                 + 1.23236 w_k1_MSFT_s002
                                 + 1.02362 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.780563 w_k1_ADM_s002
                                 + 0.79604 w_k1_AXP_s002
                                 + 0.896639 w_k1_DIS_s002
                                 + 0.821347 w_k1_FDX_s002
                                 + 1.2235 w_k1_IBM_s002
                                 + 0.897503 w_k1_INTC_s002
                                 + 0.928357 w_k1_KO_s002
                                 + 0.862572 w_k1_MAT_s002
                                 + 0.940539 w_k1_MSFT_s002
                                 + 0.825072 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.832294 w_k1_ADM_s002
                                 + 0.822007 w_k1_AXP_s002
                                 + 0.831609 w_k1_DIS_s002
                                 + 0.905121 w_k1_FDX_s002
                                 + 0.741519 w_k1_IBM_s002
                                 + 0.893426 w_k1_INTC_s002
                                 + 0.842532 w_k1_KO_s002
                                 + 0.660271 w_k1_MAT_s002
                                 + 1.08157 w_k1_MSFT_s002
                                 + 0.857003 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.2525 w_k1_ADM_s003
                                 + 1.80312 w_k1_AXP_s003
                                 + 1.56754 w_k1_DIS_s003
                                 + 1.33243 w_k1_FDX_s003
                                 + 1.38363 w_k1_IBM_s003
                                 + 1.42211 w_k1_INTC_s003
                                 + 1.38723 w_k1_KO_s003 + 1.32577 w_k1_MAT_s003
                                 + 1.22345 w_k1_MSFT_s003
                                 + 1.24895 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.873899 w_k1_ADM_s003
                                 + 1.05588 w_k1_AXP_s003
                                 + 0.859922 w_k1_DIS_s003
                                 + 1.02316 w_k1_FDX_s003
                                 + 0.781591 w_k1_IBM_s003
                                 + 1.04803 w_k1_INTC_s003
                                 + 0.865297 w_k1_KO_s003
                                 + 0.901408 w_k1_MAT_s003
                                 + 0.968193 w_k1_MSFT_s003
                                 + 1.03984 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.696496 w_k1_ADM_s003
                                 + 1.07281 w_k1_AXP_s003
                                 + 0.994174 w_k1_DIS_s003
                                 + 1.00669 w_k1_FDX_s003
                                 + 0.935629 w_k1_IBM_s003
                                 + 1.23043 w_k1_INTC_s003
                                 + 0.846652 w_k1_KO_s003
                                 + 0.880483 w_k1_MAT_s003
                                 + 1.01859 w_k1_MSFT_s003
                                 + 0.941472 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.19362 w_k1_ADM_s003
                                 + 1.3065 w_k1_AXP_s003 + 1.1641 w_k1_DIS_s003
                                 + 1.16202 w_k1_FDX_s003
                                 + 0.83505 w_k1_IBM_s003
                                 + 1.21352 w_k1_INTC_s003
                                 + 0.978159 w_k1_KO_s003
                                 + 0.962967 w_k1_MAT_s003
                                 + 1.09688 w_k1_MSFT_s003
                                 + 1.07169 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.30507 w_k1_ADM_s003
                                 + 1.47261 w_k1_AXP_s003
                                 + 1.35186 w_k1_DIS_s003
                                 + 1.18171 w_k1_FDX_s003 + 1.0028 w_k1_IBM_s003
                                 + 1.19047 w_k1_INTC_s003
                                 + 1.03648 w_k1_KO_s003
                                 + 0.905267 w_k1_MAT_s003
                                 + 1.28598 w_k1_MSFT_s003
                                 + 1.14633 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.925454 w_k1_ADM_s003
                                 + 0.966718 w_k1_AXP_s003
                                 + 0.92418 w_k1_DIS_s003
                                 + 0.832388 w_k1_FDX_s003
                                 + 1.27036 w_k1_IBM_s003
                                 + 0.801179 w_k1_INTC_s003
                                 + 1.02479 w_k1_KO_s003 + 1.20008 w_k1_MAT_s003
                                 + 0.840767 w_k1_MSFT_s003
                                 + 1.04653 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.737057 w_k1_ADM_s003
                                 + 0.790776 w_k1_AXP_s003
                                 + 0.667078 w_k1_DIS_s003
                                 + 0.787608 w_k1_FDX_s003
                                 + 0.80692 w_k1_IBM_s003
                                 + 0.967713 w_k1_INTC_s003
                                 + 0.703817 w_k1_KO_s003
                                 + 0.517386 w_k1_MAT_s003
                                 + 0.752358 w_k1_MSFT_s003
                                 + 0.834724 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.12083 w_k1_ADM_s003
                                 + 1.20938 w_k1_AXP_s003
                                 + 1.10022 w_k1_DIS_s003
                                 + 1.08089 w_k1_FDX_s003
                                 + 1.24569 w_k1_IBM_s003
                                 + 1.09346 w_k1_INTC_s003
                                 + 1.09488 w_k1_KO_s003 + 1.08302 w_k1_MAT_s003
                                 + 1.0107 w_k1_MSFT_s003 + 1.05546 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.751008 w_k1_ADM_s004
                                 + 0.802445 w_k1_AXP_s004
                                 + 0.664513 w_k1_DIS_s004
                                 + 0.835729 w_k1_FDX_s004
                                 + 0.970481 w_k1_IBM_s004
                                 + 1.00749 w_k1_INTC_s004
                                 + 0.783799 w_k1_KO_s004
                                 + 0.853898 w_k1_MAT_s004
                                 + 0.874626 w_k1_MSFT_s004
                                 + 0.876803 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.723758 w_k1_ADM_s004
                                 + 1.04963 w_k1_AXP_s004
                                 + 0.960085 w_k1_DIS_s004
                                 + 0.997616 w_k1_FDX_s004
                                 + 0.872855 w_k1_IBM_s004
                                 + 1.01567 w_k1_INTC_s004
                                 + 0.884843 w_k1_KO_s004
                                 + 0.91815 w_k1_MAT_s004
                                 + 0.898921 w_k1_MSFT_s004
                                 + 0.947867 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.628089 w_k1_ADM_s004
                                 + 0.664048 w_k1_AXP_s004
                                 + 0.557494 w_k1_DIS_s004
                                 + 0.674706 w_k1_FDX_s004
                                 + 0.718805 w_k1_IBM_s004
                                 + 0.82939 w_k1_INTC_s004
                                 + 0.626574 w_k1_KO_s004
                                 + 0.554016 w_k1_MAT_s004
                                 + 0.6627 w_k1_MSFT_s004
                                 + 0.685891 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.763226 w_k1_ADM_s004
                                 + 0.798675 w_k1_AXP_s004
                                 + 0.713715 w_k1_DIS_s004
                                 + 0.812005 w_k1_FDX_s004
                                 + 0.629482 w_k1_IBM_s004
                                 + 0.867625 w_k1_INTC_s004
                                 + 0.728492 w_k1_KO_s004
                                 + 0.617541 w_k1_MAT_s004
                                 + 0.830908 w_k1_MSFT_s004
                                 + 0.912331 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.08918 w_k1_ADM_s004
                                 + 1.37801 w_k1_AXP_s004
                                 + 1.36504 w_k1_DIS_s004
                                 + 1.11856 w_k1_FDX_s004 + 1.4918 w_k1_IBM_s004
                                 + 1.20298 w_k1_INTC_s004
                                 + 1.33736 w_k1_KO_s004 + 1.61672 w_k1_MAT_s004
                                 + 1.22993 w_k1_MSFT_s004
                                 + 1.08455 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.987531 w_k1_ADM_s004
                                 + 1.36407 w_k1_AXP_s004
                                 + 1.73107 w_k1_DIS_s004
                                 + 1.34773 w_k1_FDX_s004
                                 + 1.08874 w_k1_IBM_s004
                                 + 1.17528 w_k1_INTC_s004
                                 + 1.39516 w_k1_KO_s004 + 1.46106 w_k1_MAT_s004
                                 + 1.54159 w_k1_MSFT_s004
                                 + 1.19518 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.575474 w_k1_ADM_s004
                                 + 0.788072 w_k1_AXP_s004
                                 + 0.848966 w_k1_DIS_s004
                                 + 0.755446 w_k1_FDX_s004
                                 + 1.12448 w_k1_IBM_s004
                                 + 0.964695 w_k1_INTC_s004
                                 + 1.0026 w_k1_KO_s004 + 1.33166 w_k1_MAT_s004
                                 + 0.857193 w_k1_MSFT_s004
                                 + 0.820291 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.880159 w_k1_ADM_s004
                                 + 1.02305 w_k1_AXP_s004
                                 + 1.04817 w_k1_DIS_s004
                                 + 0.862712 w_k1_FDX_s004
                                 + 1.33113 w_k1_IBM_s004
                                 + 1.26013 w_k1_INTC_s004
                                 + 1.08938 w_k1_KO_s004
                                 + 0.837679 w_k1_MAT_s004
                                 + 0.976398 w_k1_MSFT_s004
                                 + 1.01704 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.02402 w_k1_ADM_s005
                                 + 1.00297 w_k1_AXP_s005
                                 + 0.986236 w_k1_DIS_s005
                                 + 1.09058 w_k1_FDX_s005
                                 + 0.757747 w_k1_IBM_s005
                                 + 1.17101 w_k1_INTC_s005
                                 + 0.790763 w_k1_KO_s005
                                 + 0.606537 w_k1_MAT_s005
                                 + 1.00401 w_k1_MSFT_s005 + 1.0181 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.08405 w_k1_ADM_s005
                                 + 0.960746 w_k1_AXP_s005
                                 + 0.914396 w_k1_DIS_s005
                                 + 0.895625 w_k1_FDX_s005
                                 + 1.08962 w_k1_IBM_s005
                                 + 0.900727 w_k1_INTC_s005
                                 + 0.814508 w_k1_KO_s005
                                 + 1.02135 w_k1_MAT_s005
                                 + 0.872471 w_k1_MSFT_s005
                                 + 0.987913 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.25964 w_k1_ADM_s005
                                 + 1.34405 w_k1_AXP_s005
                                 + 1.58979 w_k1_DIS_s005
                                 + 1.28259 w_k1_FDX_s005
                                 + 1.30444 w_k1_IBM_s005
                                 + 1.15909 w_k1_INTC_s005 + 1.2555 w_k1_KO_s005
                                 + 1.38835 w_k1_MAT_s005
                                 + 1.14029 w_k1_MSFT_s005 + 1.2666 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.994666 w_k1_ADM_s005
                                 + 1.22744 w_k1_AXP_s005
                                 + 1.26461 w_k1_DIS_s005 + 1.0298 w_k1_FDX_s005
                                 + 1.24732 w_k1_IBM_s005
                                 + 1.12357 w_k1_INTC_s005
                                 + 1.06356 w_k1_KO_s005 + 1.40248 w_k1_MAT_s005
                                 + 1.01989 w_k1_MSFT_s005
                                 + 1.06748 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.1705 w_k1_ADM_s005
                                 + 1.64701 w_k1_AXP_s005
                                 + 1.87197 w_k1_DIS_s005
                                 + 1.33435 w_k1_FDX_s005 + 1.7045 w_k1_IBM_s005
                                 + 1.1649 w_k1_INTC_s005 + 1.47837 w_k1_KO_s005
                                 + 1.43387 w_k1_MAT_s005
                                 + 1.26922 w_k1_MSFT_s005
                                 + 1.31796 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.918908 w_k1_ADM_s005
                                 + 0.782321 w_k1_AXP_s005
                                 + 0.706177 w_k1_DIS_s005
                                 + 0.877663 w_k1_FDX_s005
                                 + 0.647075 w_k1_IBM_s005
                                 + 0.947952 w_k1_INTC_s005
                                 + 0.693659 w_k1_KO_s005
                                 + 0.726202 w_k1_MAT_s005
                                 + 0.825236 w_k1_MSFT_s005
                                 + 0.812942 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.76717 w_k1_ADM_s005
                                 + 0.681754 w_k1_AXP_s005
                                 + 0.86833 w_k1_DIS_s005
                                 + 0.874136 w_k1_FDX_s005
                                 + 0.882986 w_k1_IBM_s005
                                 + 0.946014 w_k1_INTC_s005
                                 + 0.847993 w_k1_KO_s005
                                 + 0.619638 w_k1_MAT_s005
                                 + 0.969355 w_k1_MSFT_s005
                                 + 0.969753 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.742233 w_k1_ADM_s005
                                 + 0.741811 w_k1_AXP_s005
                                 + 0.63116 w_k1_DIS_s005
                                 + 0.712056 w_k1_FDX_s005
                                 + 0.824984 w_k1_IBM_s005
                                 + 0.823598 w_k1_INTC_s005
                                 + 0.711818 w_k1_KO_s005
                                 + 0.795413 w_k1_MAT_s005
                                 + 0.690524 w_k1_MSFT_s005
                                 + 0.826139 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.06062 w_k1_ADM_s006
                                 + 1.17738 w_k1_AXP_s006
                                 + 1.08796 w_k1_DIS_s006
                                 + 1.10493 w_k1_FDX_s006
                                 + 1.15594 w_k1_IBM_s006
                                 + 0.992664 w_k1_INTC_s006
                                 + 1.10814 w_k1_KO_s006 + 1.82898 w_k1_MAT_s006
                                 + 1.08544 w_k1_MSFT_s006
                                 + 1.09772 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.06346 w_k1_ADM_s006
                                 + 0.8017 w_k1_AXP_s006 + 1.15212 w_k1_DIS_s006
                                 + 0.978959 w_k1_FDX_s006
                                 + 0.968152 w_k1_IBM_s006
                                 + 1.12542 w_k1_INTC_s006
                                 + 1.20115 w_k1_KO_s006
                                 + 0.892947 w_k1_MAT_s006
                                 + 1.13724 w_k1_MSFT_s006
                                 + 1.09278 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.41796 w_k1_ADM_s006
                                 + 1.6861 w_k1_AXP_s006 + 1.91593 w_k1_DIS_s006
                                 + 1.52844 w_k1_FDX_s006
                                 + 1.44792 w_k1_IBM_s006
                                 + 1.53645 w_k1_INTC_s006
                                 + 1.38262 w_k1_KO_s006 + 1.74967 w_k1_MAT_s006
                                 + 1.51998 w_k1_MSFT_s006
                                 + 1.35732 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.872333 w_k1_ADM_s006
                                 + 0.641047 w_k1_AXP_s006
                                 + 0.645932 w_k1_DIS_s006
                                 + 0.763449 w_k1_FDX_s006
                                 + 0.71577 w_k1_IBM_s006
                                 + 0.869029 w_k1_INTC_s006
                                 + 0.72592 w_k1_KO_s006
                                 + 0.671995 w_k1_MAT_s006
                                 + 0.77587 w_k1_MSFT_s006
                                 + 0.914335 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.822636 w_k1_ADM_s006
                                 + 0.892989 w_k1_AXP_s006
                                 + 0.841031 w_k1_DIS_s006
                                 + 0.827625 w_k1_FDX_s006
                                 + 1.33308 w_k1_IBM_s006
                                 + 1.16666 w_k1_INTC_s006
                                 + 1.06901 w_k1_KO_s006 + 1.70212 w_k1_MAT_s006
                                 + 1.05502 w_k1_MSFT_s006
                                 + 1.02331 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.822462 w_k1_ADM_s006
                                 + 0.777073 w_k1_AXP_s006
                                 + 0.804381 w_k1_DIS_s006
                                 + 0.907997 w_k1_FDX_s006
                                 + 0.927502 w_k1_IBM_s006
                                 + 1.04733 w_k1_INTC_s006
                                 + 1.02581 w_k1_KO_s006 + 1.29314 w_k1_MAT_s006
                                 + 1.01728 w_k1_MSFT_s006
                                 + 0.952241 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.22369 w_k1_ADM_s006
                                 + 1.05047 w_k1_AXP_s006
                                 + 1.29419 w_k1_DIS_s006 + 1.1903 w_k1_FDX_s006
                                 + 0.951562 w_k1_IBM_s006
                                 + 1.30459 w_k1_INTC_s006
                                 + 1.28865 w_k1_KO_s006 + 1.23255 w_k1_MAT_s006
                                 + 1.25054 w_k1_MSFT_s006
                                 + 1.29454 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.846122 w_k1_ADM_s006
                                 + 0.829358 w_k1_AXP_s006
                                 + 0.529915 w_k1_DIS_s006
                                 + 0.761478 w_k1_FDX_s006
                                 + 0.77515 w_k1_IBM_s006
                                 + 0.813386 w_k1_INTC_s006
                                 + 0.679201 w_k1_KO_s006
                                 + 0.878832 w_k1_MAT_s006
                                 + 0.715646 w_k1_MSFT_s006
                                 + 0.93129 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.95725 w_k1_ADM_s007
                                 + 0.930955 w_k1_AXP_s007
                                 + 0.708167 w_k1_DIS_s007
                                 + 0.772152 w_k1_FDX_s007
                                 + 0.949867 w_k1_IBM_s007
                                 + 0.863407 w_k1_INTC_s007
                                 + 0.770532 w_k1_KO_s007
                                 + 0.894336 w_k1_MAT_s007
                                 + 0.756025 w_k1_MSFT_s007
                                 + 0.755089 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.22037 w_k1_ADM_s007
                                 + 1.81052 w_k1_AXP_s007
                                 + 1.60855 w_k1_DIS_s007
                                 + 1.45676 w_k1_FDX_s007
                                 + 1.30588 w_k1_IBM_s007
                                 + 1.65968 w_k1_INTC_s007
                                 + 1.32273 w_k1_KO_s007 + 1.59325 w_k1_MAT_s007
                                 + 1.31084 w_k1_MSFT_s007
                                 + 1.18101 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.7267 w_k1_ADM_s007
                                 + 0.829704 w_k1_AXP_s007
                                 + 1.15965 w_k1_DIS_s007
                                 + 1.02562 w_k1_FDX_s007
                                 + 0.918156 w_k1_IBM_s007
                                 + 1.26324 w_k1_INTC_s007
                                 + 1.01425 w_k1_KO_s007
                                 + 0.641309 w_k1_MAT_s007
                                 + 1.1471 w_k1_MSFT_s007
                                 + 0.957009 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.04897 w_k1_ADM_s007
                                 + 1.4975 w_k1_AXP_s007 + 1.00019 w_k1_DIS_s007
                                 + 1.18019 w_k1_FDX_s007
                                 + 0.708796 w_k1_IBM_s007
                                 + 1.2433 w_k1_INTC_s007
                                 + 0.810421 w_k1_KO_s007
                                 + 0.913986 w_k1_MAT_s007
                                 + 0.964126 w_k1_MSFT_s007
                                 + 1.03732 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.17447 w_k1_ADM_s007
                                 + 1.31321 w_k1_AXP_s007
                                 + 1.11376 w_k1_DIS_s007
                                 + 1.08743 w_k1_FDX_s007
                                 + 1.05152 w_k1_IBM_s007
                                 + 1.31361 w_k1_INTC_s007
                                 + 1.14927 w_k1_KO_s007 + 1.34555 w_k1_MAT_s007
                                 + 1.0716 w_k1_MSFT_s007 + 1.16162 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.0621 w_k1_ADM_s007
                                 + 1.32366 w_k1_AXP_s007 + 1.3976 w_k1_DIS_s007
                                 + 1.22548 w_k1_FDX_s007
                                 + 1.09696 w_k1_IBM_s007
                                 + 1.40442 w_k1_INTC_s007 + 1.1545 w_k1_KO_s007
                                 + 0.950143 w_k1_MAT_s007
                                 + 1.10484 w_k1_MSFT_s007
                                 + 1.15677 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.844924 w_k1_ADM_s007
                                 + 0.971957 w_k1_AXP_s007
                                 + 0.795747 w_k1_DIS_s007
                                 + 0.885264 w_k1_FDX_s007
                                 + 0.722217 w_k1_IBM_s007
                                 + 0.86026 w_k1_INTC_s007
                                 + 0.659501 w_k1_KO_s007
                                 + 0.78752 w_k1_MAT_s007
                                 + 0.863502 w_k1_MSFT_s007
                                 + 0.867733 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.12296 w_k1_ADM_s007
                                 + 0.925884 w_k1_AXP_s007
                                 + 1.04029 w_k1_DIS_s007
                                 + 0.912018 w_k1_FDX_s007
                                 + 1.30398 w_k1_IBM_s007
                                 + 1.10397 w_k1_INTC_s007
                                 + 1.00907 w_k1_KO_s007
                                 + 0.846076 w_k1_MAT_s007
                                 + 0.874228 w_k1_MSFT_s007
                                 + 1.03325 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.960727 w_k1_ADM_s010
                                 + 1.06587 w_k1_AXP_s010
                                 + 1.04317 w_k1_DIS_s010
                                 + 1.20855 w_k1_FDX_s010
                                 + 0.914543 w_k1_IBM_s010
                                 + 1.01221 w_k1_INTC_s010
                                 + 1.01262 w_k1_KO_s010
                                 + 0.860899 w_k1_MAT_s010
                                 + 1.1248 w_k1_MSFT_s010 + 1.06912 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.787185 w_k1_ADM_s010
                                 + 0.659083 w_k1_AXP_s010
                                 + 0.82121 w_k1_DIS_s010
                                 + 0.822253 w_k1_FDX_s010
                                 + 0.673975 w_k1_IBM_s010
                                 + 0.906881 w_k1_INTC_s010
                                 + 0.756349 w_k1_KO_s010
                                 + 0.433456 w_k1_MAT_s010
                                 + 0.915989 w_k1_MSFT_s010
                                 + 0.769923 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.83597 w_k1_ADM_s010
                                 + 0.962691 w_k1_AXP_s010
                                 + 0.833072 w_k1_DIS_s010
                                 + 0.890466 w_k1_FDX_s010
                                 + 0.882184 w_k1_IBM_s010
                                 + 0.700949 w_k1_INTC_s010
                                 + 0.786507 w_k1_KO_s010
                                 + 0.797825 w_k1_MAT_s010
                                 + 0.781808 w_k1_MSFT_s010
                                 + 0.93077 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.637225 w_k1_ADM_s010
                                 + 0.854139 w_k1_AXP_s010
                                 + 0.864028 w_k1_DIS_s010
                                 + 0.812608 w_k1_FDX_s010
                                 + 1.25939 w_k1_IBM_s010
                                 + 0.727804 w_k1_INTC_s010
                                 + 0.929737 w_k1_KO_s010
                                 + 1.04666 w_k1_MAT_s010
                                 + 0.753377 w_k1_MSFT_s010
                                 + 0.885478 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.11749 w_k1_ADM_s010
                                 + 1.16679 w_k1_AXP_s010
                                 + 1.45396 w_k1_DIS_s010
                                 + 1.34435 w_k1_FDX_s010
                                 + 0.927475 w_k1_IBM_s010
                                 + 1.16476 w_k1_INTC_s010
                                 + 1.07499 w_k1_KO_s010 + 1.17651 w_k1_MAT_s010
                                 + 1.34945 w_k1_MSFT_s010
                                 + 1.10397 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.947009 w_k1_ADM_s010
                                 + 1.46863 w_k1_AXP_s010
                                 + 1.23317 w_k1_DIS_s010
                                 + 1.18421 w_k1_FDX_s010
                                 + 1.88509 w_k1_IBM_s010
                                 + 1.19619 w_k1_INTC_s010
                                 + 1.41373 w_k1_KO_s010 + 2.28524 w_k1_MAT_s010
                                 + 1.2952 w_k1_MSFT_s010 + 1.16636 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.821683 w_k1_ADM_s010
                                 + 0.952984 w_k1_AXP_s010
                                 + 1.04107 w_k1_DIS_s010
                                 + 0.910924 w_k1_FDX_s010
                                 + 1.19473 w_k1_IBM_s010
                                 + 0.894937 w_k1_INTC_s010
                                 + 0.88979 w_k1_KO_s010 + 1.16458 w_k1_MAT_s010
                                 + 0.991536 w_k1_MSFT_s010
                                 + 0.91135 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04391 w_k1_ADM_s010
                                 + 1.35038 w_k1_AXP_s010
                                 + 1.46657 w_k1_DIS_s010
                                 + 1.15835 w_k1_FDX_s010
                                 + 1.35378 w_k1_IBM_s010
                                 + 0.939346 w_k1_INTC_s010
                                 + 1.18515 w_k1_KO_s010 + 1.16938 w_k1_MAT_s010
                                 + 1.04867 w_k1_MSFT_s010 + 1.1683 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.17037 w_k1_ADM_s011
                                 + 1.01978 w_k1_AXP_s011
                                 + 1.08694 w_k1_DIS_s011
                                 + 1.15352 w_k1_FDX_s011
                                 + 0.917131 w_k1_IBM_s011
                                 + 1.10718 w_k1_INTC_s011
                                 + 1.10192 w_k1_KO_s011 + 1.332 w_k1_MAT_s011
                                 + 1.21364 w_k1_MSFT_s011
                                 + 1.18801 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.13721 w_k1_ADM_s011
                                 + 0.903811 w_k1_AXP_s011
                                 + 1.34883 w_k1_DIS_s011
                                 + 1.08853 w_k1_FDX_s011
                                 + 1.07759 w_k1_IBM_s011
                                 + 1.06013 w_k1_INTC_s011
                                 + 1.31834 w_k1_KO_s011 + 1.09067 w_k1_MAT_s011
                                 + 1.19648 w_k1_MSFT_s011
                                 + 1.18015 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.5883 w_k1_ADM_s011
                                 + 1.63407 w_k1_AXP_s011
                                 + 1.95304 w_k1_DIS_s011
                                 + 1.61648 w_k1_FDX_s011
                                 + 1.39918 w_k1_IBM_s011
                                 + 1.21949 w_k1_INTC_s011
                                 + 1.43661 w_k1_KO_s011 + 2.06751 w_k1_MAT_s011
                                 + 1.38933 w_k1_MSFT_s011
                                 + 1.18699 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.24518 w_k1_ADM_s011
                                 + 1.04868 w_k1_AXP_s011
                                 + 1.45938 w_k1_DIS_s011
                                 + 1.03514 w_k1_FDX_s011
                                 + 1.72107 w_k1_IBM_s011
                                 + 0.957231 w_k1_INTC_s011
                                 + 1.50114 w_k1_KO_s011 + 1.67313 w_k1_MAT_s011
                                 + 1.16077 w_k1_MSFT_s011
                                 + 1.14164 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.30221 w_k1_ADM_s011
                                 + 1.4547 w_k1_AXP_s011 + 1.20006 w_k1_DIS_s011
                                 + 1.13101 w_k1_FDX_s011
                                 + 1.66177 w_k1_IBM_s011
                                 + 1.26241 w_k1_INTC_s011
                                 + 1.37889 w_k1_KO_s011 + 2.08971 w_k1_MAT_s011
                                 + 1.14896 w_k1_MSFT_s011
                                 + 1.21461 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.893156 w_k1_ADM_s011
                                 + 1.05397 w_k1_AXP_s011
                                 + 0.660199 w_k1_DIS_s011
                                 + 0.820607 w_k1_FDX_s011
                                 + 1.11184 w_k1_IBM_s011
                                 + 0.902309 w_k1_INTC_s011
                                 + 0.838023 w_k1_KO_s011
                                 + 1.35691 w_k1_MAT_s011
                                 + 0.764931 w_k1_MSFT_s011
                                 + 0.874939 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.821768 w_k1_ADM_s011
                                 + 0.77898 w_k1_AXP_s011
                                 + 0.953105 w_k1_DIS_s011
                                 + 0.860865 w_k1_FDX_s011
                                 + 0.917775 w_k1_IBM_s011
                                 + 1.12339 w_k1_INTC_s011
                                 + 0.956441 w_k1_KO_s011
                                 + 0.856387 w_k1_MAT_s011
                                 + 1.0765 w_k1_MSFT_s011
                                 + 0.909428 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.824145 w_k1_ADM_s011
                                 + 0.72019 w_k1_AXP_s011
                                 + 0.725889 w_k1_DIS_s011
                                 + 0.803533 w_k1_FDX_s011
                                 + 0.818774 w_k1_IBM_s011
                                 + 0.917035 w_k1_INTC_s011
                                 + 0.784191 w_k1_KO_s011
                                 + 0.782885 w_k1_MAT_s011
                                 + 0.871524 w_k1_MSFT_s011
                                 + 0.90816 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.921203 w_k1_ADM_s012
                                 + 1.03868 w_k1_AXP_s012
                                 + 0.857823 w_k1_DIS_s012
                                 + 0.826754 w_k1_FDX_s012
                                 + 1.28482 w_k1_IBM_s012
                                 + 0.962084 w_k1_INTC_s012
                                 + 0.997214 w_k1_KO_s012
                                 + 1.54394 w_k1_MAT_s012
                                 + 0.963433 w_k1_MSFT_s012
                                 + 0.911921 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.8116 w_k1_ADM_s012
                                 + 1.72053 w_k1_AXP_s012
                                 + 1.63454 w_k1_DIS_s012
                                 + 1.44956 w_k1_FDX_s012
                                 + 1.44227 w_k1_IBM_s012
                                 + 1.60263 w_k1_INTC_s012 + 1.5203 w_k1_KO_s012
                                 + 2.17925 w_k1_MAT_s012
                                 + 1.26312 w_k1_MSFT_s012 + 1.4233 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.01486 w_k1_ADM_s012
                                 + 1.04111 w_k1_AXP_s012
                                 + 1.08789 w_k1_DIS_s012
                                 + 0.977306 w_k1_FDX_s012
                                 + 1.08794 w_k1_IBM_s012
                                 + 1.32433 w_k1_INTC_s012
                                 + 1.02962 w_k1_KO_s012
                                 + 0.955173 w_k1_MAT_s012
                                 + 1.10663 w_k1_MSFT_s012
                                 + 1.16456 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.11034 w_k1_ADM_s012
                                 + 1.22218 w_k1_AXP_s012
                                 + 0.919363 w_k1_DIS_s012
                                 + 1.11082 w_k1_FDX_s012
                                 + 0.869132 w_k1_IBM_s012
                                 + 1.11821 w_k1_INTC_s012
                                 + 0.865606 w_k1_KO_s012
                                 + 1.03869 w_k1_MAT_s012
                                 + 1.04656 w_k1_MSFT_s012
                                 + 1.20618 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.30103 w_k1_ADM_s012
                                 + 1.42204 w_k1_AXP_s012
                                 + 1.53883 w_k1_DIS_s012
                                 + 1.44919 w_k1_FDX_s012
                                 + 0.840592 w_k1_IBM_s012
                                 + 1.44944 w_k1_INTC_s012
                                 + 1.09971 w_k1_KO_s012
                                 + 0.941853 w_k1_MAT_s012
                                 + 1.31279 w_k1_MSFT_s012
                                 + 1.19789 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.23235 w_k1_ADM_s012
                                 + 1.10272 w_k1_AXP_s012
                                 + 1.40705 w_k1_DIS_s012
                                 + 1.05853 w_k1_FDX_s012
                                 + 1.29401 w_k1_IBM_s012
                                 + 1.27582 w_k1_INTC_s012 + 1.235 w_k1_KO_s012
                                 + 1.04404 w_k1_MAT_s012
                                 + 1.24619 w_k1_MSFT_s012
                                 + 1.06685 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.858587 w_k1_ADM_s012
                                 + 0.671996 w_k1_AXP_s012
                                 + 0.701009 w_k1_DIS_s012
                                 + 0.743221 w_k1_FDX_s012
                                 + 0.929161 w_k1_IBM_s012
                                 + 0.87179 w_k1_INTC_s012
                                 + 0.786939 w_k1_KO_s012
                                 + 0.931609 w_k1_MAT_s012
                                 + 0.822331 w_k1_MSFT_s012
                                 + 0.953423 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.897164 w_k1_ADM_s012
                                 + 0.706556 w_k1_AXP_s012
                                 + 0.816847 w_k1_DIS_s012
                                 + 0.925463 w_k1_FDX_s012
                                 + 0.732141 w_k1_IBM_s012
                                 + 1.07891 w_k1_INTC_s012
                                 + 0.851396 w_k1_KO_s012
                                 + 0.63432 w_k1_MAT_s012
                                 + 1.06198 w_k1_MSFT_s012
                                 + 1.01891 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.999026 w_k1_ADM_s013
                                 + 1.08532 w_k1_AXP_s013
                                 + 0.833536 w_k1_DIS_s013
                                 + 0.916173 w_k1_FDX_s013
                                 + 1.26916 w_k1_IBM_s013
                                 + 0.752464 w_k1_INTC_s013
                                 + 1.0823 w_k1_KO_s013 + 1.37123 w_k1_MAT_s013
                                 + 0.854507 w_k1_MSFT_s013
                                 + 0.976669 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.30319 w_k1_ADM_s013
                                 + 1.39982 w_k1_AXP_s013
                                 + 1.16173 w_k1_DIS_s013
                                 + 1.22392 w_k1_FDX_s013
                                 + 1.08026 w_k1_IBM_s013
                                 + 1.02735 w_k1_INTC_s013
                                 + 1.11029 w_k1_KO_s013 + 1.23344 w_k1_MAT_s013
                                 + 1.15492 w_k1_MSFT_s013
                                 + 1.05257 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.918342 w_k1_ADM_s013
                                 + 1.3289 w_k1_AXP_s013 + 1.34345 w_k1_DIS_s013
                                 + 1.07182 w_k1_FDX_s013
                                 + 2.05016 w_k1_IBM_s013
                                 + 1.05732 w_k1_INTC_s013
                                 + 1.61141 w_k1_KO_s013 + 1.54771 w_k1_MAT_s013
                                 + 1.02777 w_k1_MSFT_s013
                                 + 1.18422 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.0583 w_k1_ADM_s013
                                 + 0.862506 w_k1_AXP_s013
                                 + 0.748952 w_k1_DIS_s013
                                 + 0.970637 w_k1_FDX_s013
                                 + 0.645774 w_k1_IBM_s013
                                 + 0.78788 w_k1_INTC_s013
                                 + 0.905067 w_k1_KO_s013
                                 + 0.727005 w_k1_MAT_s013
                                 + 0.977421 w_k1_MSFT_s013
                                 + 1.05643 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.23009 w_k1_ADM_s013
                                 + 0.86537 w_k1_AXP_s013
                                 + 1.02252 w_k1_DIS_s013
                                 + 0.984486 w_k1_FDX_s013
                                 + 1.17611 w_k1_IBM_s013
                                 + 0.736755 w_k1_INTC_s013
                                 + 1.14417 w_k1_KO_s013
                                 + 0.984849 w_k1_MAT_s013
                                 + 0.924885 w_k1_MSFT_s013
                                 + 0.965833 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11233 w_k1_ADM_s013
                                 + 0.795138 w_k1_AXP_s013
                                 + 0.747665 w_k1_DIS_s013
                                 + 0.857158 w_k1_FDX_s013
                                 + 0.879438 w_k1_IBM_s013
                                 + 0.634599 w_k1_INTC_s013
                                 + 0.923852 w_k1_KO_s013
                                 + 0.643976 w_k1_MAT_s013
                                 + 0.741891 w_k1_MSFT_s013
                                 + 0.920847 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.23633 w_k1_ADM_s013
                                 + 1.25264 w_k1_AXP_s013
                                 + 1.63343 w_k1_DIS_s013 + 1.2158 w_k1_FDX_s013
                                 + 1.08579 w_k1_IBM_s013
                                 + 1.05662 w_k1_INTC_s013
                                 + 1.39665 w_k1_KO_s013 + 1.0763 w_k1_MAT_s013
                                 + 1.27817 w_k1_MSFT_s013
                                 + 1.19279 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.818274 w_k1_ADM_s013
                                 + 0.693442 w_k1_AXP_s013
                                 + 0.751674 w_k1_DIS_s013
                                 + 0.74249 w_k1_FDX_s013 + 1.0008 w_k1_IBM_s013
                                 + 0.786193 w_k1_INTC_s013
                                 + 0.987287 w_k1_KO_s013
                                 + 0.782893 w_k1_MAT_s013
                                 + 0.864208 w_k1_MSFT_s013
                                 + 1.00939 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.868515 w_k1_ADM_s014
                                 + 0.748711 w_k1_AXP_s014
                                 + 0.5555 w_k1_DIS_s014
                                 + 0.832893 w_k1_FDX_s014
                                 + 0.956029 w_k1_IBM_s014
                                 + 0.792364 w_k1_INTC_s014
                                 + 0.735935 w_k1_KO_s014
                                 + 1.06321 w_k1_MAT_s014
                                 + 0.748124 w_k1_MSFT_s014
                                 + 0.887206 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.37355 w_k1_ADM_s014
                                 + 0.89993 w_k1_AXP_s014
                                 + 1.10614 w_k1_DIS_s014
                                 + 1.03794 w_k1_FDX_s014
                                 + 0.757621 w_k1_IBM_s014
                                 + 1.00154 w_k1_INTC_s014
                                 + 0.931301 w_k1_KO_s014
                                 + 0.886141 w_k1_MAT_s014
                                 + 1.07752 w_k1_MSFT_s014
                                 + 0.981371 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.44192 w_k1_ADM_s014
                                 + 1.24185 w_k1_AXP_s014
                                 + 0.87542 w_k1_DIS_s014
                                 + 1.09985 w_k1_FDX_s014
                                 + 0.80034 w_k1_IBM_s014
                                 + 1.10944 w_k1_INTC_s014
                                 + 0.831801 w_k1_KO_s014
                                 + 0.970663 w_k1_MAT_s014
                                 + 0.922552 w_k1_MSFT_s014
                                 + 0.927235 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.32592 w_k1_ADM_s014
                                 + 1.20611 w_k1_AXP_s014
                                 + 1.07504 w_k1_DIS_s014
                                 + 1.09415 w_k1_FDX_s014
                                 + 1.50888 w_k1_IBM_s014
                                 + 0.90398 w_k1_INTC_s014
                                 + 1.28642 w_k1_KO_s014 + 1.854 w_k1_MAT_s014
                                 + 1.00875 w_k1_MSFT_s014
                                 + 1.08439 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.07496 w_k1_ADM_s014
                                 + 0.760399 w_k1_AXP_s014
                                 + 0.874534 w_k1_DIS_s014
                                 + 0.80854 w_k1_FDX_s014
                                 + 1.05622 w_k1_IBM_s014
                                 + 0.877956 w_k1_INTC_s014
                                 + 0.902784 w_k1_KO_s014
                                 + 0.879579 w_k1_MAT_s014
                                 + 0.855347 w_k1_MSFT_s014
                                 + 0.973995 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.821212 w_k1_ADM_s014
                                 + 1.00665 w_k1_AXP_s014
                                 + 0.798783 w_k1_DIS_s014
                                 + 0.879584 w_k1_FDX_s014
                                 + 0.791706 w_k1_IBM_s014
                                 + 0.799515 w_k1_INTC_s014
                                 + 0.814775 w_k1_KO_s014
                                 + 0.844383 w_k1_MAT_s014
                                 + 0.797681 w_k1_MSFT_s014
                                 + 0.902217 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.07557 w_k1_ADM_s014
                                 + 0.690059 w_k1_AXP_s014
                                 + 0.615794 w_k1_DIS_s014
                                 + 0.789895 w_k1_FDX_s014
                                 + 0.608782 w_k1_IBM_s014
                                 + 0.752593 w_k1_INTC_s014
                                 + 0.555037 w_k1_KO_s014
                                 + 0.491805 w_k1_MAT_s014
                                 + 0.717403 w_k1_MSFT_s014
                                 + 0.717104 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.38078 w_k1_ADM_s014
                                 + 1.21847 w_k1_AXP_s014
                                 + 1.26017 w_k1_DIS_s014
                                 + 1.24398 w_k1_FDX_s014 + 1.085 w_k1_IBM_s014
                                 + 1.0501 w_k1_INTC_s014
                                 + 0.998273 w_k1_KO_s014
                                 + 1.13433 w_k1_MAT_s014
                                 + 1.09566 w_k1_MSFT_s014
                                 + 1.08095 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.77793 w_k1_ADM_s015
                                 + 0.809011 w_k1_AXP_s015
                                 + 0.741846 w_k1_DIS_s015
                                 + 0.814124 w_k1_FDX_s015
                                 + 0.814775 w_k1_IBM_s015
                                 + 1.14554 w_k1_INTC_s015
                                 + 0.804437 w_k1_KO_s015
                                 + 0.72715 w_k1_MAT_s015
                                 + 0.963253 w_k1_MSFT_s015
                                 + 0.872571 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.16515 w_k1_ADM_s015
                                 + 1.01332 w_k1_AXP_s015
                                 + 0.930429 w_k1_DIS_s015
                                 + 1.05733 w_k1_FDX_s015
                                 + 0.712086 w_k1_IBM_s015
                                 + 1.16097 w_k1_INTC_s015
                                 + 0.832869 w_k1_KO_s015
                                 + 0.55286 w_k1_MAT_s015
                                 + 0.888678 w_k1_MSFT_s015
                                 + 0.930495 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.841777 w_k1_ADM_s015
                                 + 0.664599 w_k1_AXP_s015
                                 + 0.471926 w_k1_DIS_s015
                                 + 0.729754 w_k1_FDX_s015
                                 + 0.99647 w_k1_IBM_s015
                                 + 0.854316 w_k1_INTC_s015
                                 + 0.760352 w_k1_KO_s015
                                 + 0.739811 w_k1_MAT_s015
                                 + 0.741452 w_k1_MSFT_s015
                                 + 0.827189 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.02196 w_k1_ADM_s015
                                 + 0.951649 w_k1_AXP_s015
                                 + 0.927482 w_k1_DIS_s015
                                 + 0.974037 w_k1_FDX_s015
                                 + 1.1047 w_k1_IBM_s015
                                 + 1.03907 w_k1_INTC_s015
                                 + 1.06976 w_k1_KO_s015 + 1.24775 w_k1_MAT_s015
                                 + 0.896462 w_k1_MSFT_s015
                                 + 0.872261 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.28142 w_k1_ADM_s015
                                 + 1.20854 w_k1_AXP_s015
                                 + 1.14807 w_k1_DIS_s015
                                 + 1.02158 w_k1_FDX_s015
                                 + 1.49225 w_k1_IBM_s015
                                 + 0.848652 w_k1_INTC_s015
                                 + 1.15273 w_k1_KO_s015 + 1.47609 w_k1_MAT_s015
                                 + 0.891553 w_k1_MSFT_s015
                                 + 1.09617 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.661965 w_k1_ADM_s015
                                 + 0.788404 w_k1_AXP_s015
                                 + 0.693354 w_k1_DIS_s015
                                 + 0.830482 w_k1_FDX_s015
                                 + 0.529316 w_k1_IBM_s015
                                 + 0.905172 w_k1_INTC_s015
                                 + 0.57508 w_k1_KO_s015
                                 + 0.533494 w_k1_MAT_s015
                                 + 0.692322 w_k1_MSFT_s015
                                 + 0.762058 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.33178 w_k1_ADM_s015
                                 + 1.29444 w_k1_AXP_s015 + 1.0341 w_k1_DIS_s015
                                 + 1.14622 w_k1_FDX_s015
                                 + 1.05541 w_k1_IBM_s015
                                 + 1.18428 w_k1_INTC_s015
                                 + 1.07617 w_k1_KO_s015 + 1.12 w_k1_MAT_s015
                                 + 0.867744 w_k1_MSFT_s015
                                 + 1.06199 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.16605 w_k1_ADM_s015
                                 + 1.04923 w_k1_AXP_s015
                                 + 1.32444 w_k1_DIS_s015
                                 + 1.37522 w_k1_FDX_s015
                                 + 0.963637 w_k1_IBM_s015
                                 + 1.28877 w_k1_INTC_s015
                                 + 1.21059 w_k1_KO_s015
                                 + 0.975113 w_k1_MAT_s015
                                 + 1.23765 w_k1_MSFT_s015
                                 + 1.01444 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02123 w_k1_ADM_s016
                                 + 1.19386 w_k1_AXP_s016
                                 + 1.25171 w_k1_DIS_s016
                                 + 1.09375 w_k1_FDX_s016
                                 + 1.52015 w_k1_IBM_s016
                                 + 1.04638 w_k1_INTC_s016
                                 + 1.08809 w_k1_KO_s016 + 1.10653 w_k1_MAT_s016
                                 + 1.05977 w_k1_MSFT_s016
                                 + 1.05417 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.934195 w_k1_ADM_s016
                                 + 1.03682 w_k1_AXP_s016
                                 + 1.41322 w_k1_DIS_s016
                                 + 1.26159 w_k1_FDX_s016
                                 + 0.872324 w_k1_IBM_s016
                                 + 1.16276 w_k1_INTC_s016
                                 + 0.966124 w_k1_KO_s016
                                 + 0.678608 w_k1_MAT_s016
                                 + 1.2851 w_k1_MSFT_s016 + 1.0182 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.817819 w_k1_ADM_s016
                                 + 1.0893 w_k1_AXP_s016 + 1.14191 w_k1_DIS_s016
                                 + 0.985216 w_k1_FDX_s016
                                 + 1.07341 w_k1_IBM_s016
                                 + 0.905422 w_k1_INTC_s016
                                 + 1.04905 w_k1_KO_s016
                                 + 0.767087 w_k1_MAT_s016
                                 + 0.924756 w_k1_MSFT_s016
                                 + 0.974192 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.909251 w_k1_ADM_s016
                                 + 0.804846 w_k1_AXP_s016
                                 + 0.7759 w_k1_DIS_s016 + 0.86594 w_k1_FDX_s016
                                 + 0.957878 w_k1_IBM_s016
                                 + 0.883482 w_k1_INTC_s016
                                 + 0.869451 w_k1_KO_s016
                                 + 0.691917 w_k1_MAT_s016
                                 + 0.953238 w_k1_MSFT_s016
                                 + 0.966261 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.28115 w_k1_ADM_s016
                                 + 0.929985 w_k1_AXP_s016
                                 + 1.08904 w_k1_DIS_s016
                                 + 1.14184 w_k1_FDX_s016
                                 + 0.794494 w_k1_IBM_s016
                                 + 1.13269 w_k1_INTC_s016
                                 + 0.965491 w_k1_KO_s016
                                 + 0.620228 w_k1_MAT_s016
                                 + 1.23941 w_k1_MSFT_s016
                                 + 1.08443 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.31497 w_k1_ADM_s016
                                 + 1.24532 w_k1_AXP_s016
                                 + 1.30602 w_k1_DIS_s016
                                 + 1.31068 w_k1_FDX_s016
                                 + 1.25033 w_k1_IBM_s016
                                 + 1.11679 w_k1_INTC_s016
                                 + 1.07827 w_k1_KO_s016 + 1.06947 w_k1_MAT_s016
                                 + 1.13108 w_k1_MSFT_s016
                                 + 1.11606 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.878846 w_k1_ADM_s016
                                 + 0.809581 w_k1_AXP_s016
                                 + 0.822918 w_k1_DIS_s016
                                 + 0.902611 w_k1_FDX_s016
                                 + 0.694686 w_k1_IBM_s016
                                 + 0.88124 w_k1_INTC_s016
                                 + 0.696663 w_k1_KO_s016
                                 + 0.552661 w_k1_MAT_s016
                                 + 0.872713 w_k1_MSFT_s016
                                 + 0.893446 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.863977 w_k1_ADM_s016
                                 + 1.14505 w_k1_AXP_s016
                                 + 0.869327 w_k1_DIS_s016
                                 + 0.93283 w_k1_FDX_s016
                                 + 1.21417 w_k1_IBM_s016
                                 + 0.811192 w_k1_INTC_s016
                                 + 0.922094 w_k1_KO_s016
                                 + 1.02051 w_k1_MAT_s016
                                 + 0.889658 w_k1_MSFT_s016
                                 + 0.970134 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.949484 w_k1_ADM_s017
                                 + 0.977515 w_k1_AXP_s017
                                 + 0.841239 w_k1_DIS_s017
                                 + 0.804919 w_k1_FDX_s017
                                 + 1.29646 w_k1_IBM_s017
                                 + 0.978345 w_k1_INTC_s017
                                 + 1.12141 w_k1_KO_s017 + 1.41564 w_k1_MAT_s017
                                 + 0.82635 w_k1_MSFT_s017
                                 + 0.897733 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.888731 w_k1_ADM_s017
                                 + 0.777602 w_k1_AXP_s017
                                 + 0.68443 w_k1_DIS_s017
                                 + 0.821541 w_k1_FDX_s017
                                 + 0.638012 w_k1_IBM_s017
                                 + 0.749489 w_k1_INTC_s017
                                 + 0.840744 w_k1_KO_s017
                                 + 0.887452 w_k1_MAT_s017
                                 + 0.750395 w_k1_MSFT_s017
                                 + 0.847013 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.19191 w_k1_ADM_s017
                                 + 0.891693 w_k1_AXP_s017
                                 + 1.11563 w_k1_DIS_s017
                                 + 1.01237 w_k1_FDX_s017
                                 + 0.881718 w_k1_IBM_s017
                                 + 0.890005 w_k1_INTC_s017
                                 + 1.15445 w_k1_KO_s017
                                 + 0.831188 w_k1_MAT_s017
                                 + 0.9702 w_k1_MSFT_s017 + 0.99662 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.74183 w_k1_ADM_s017
                                 + 1.48349 w_k1_AXP_s017
                                 + 1.35084 w_k1_DIS_s017
                                 + 1.26649 w_k1_FDX_s017
                                 + 1.14142 w_k1_IBM_s017
                                 + 0.963643 w_k1_INTC_s017
                                 + 1.14964 w_k1_KO_s017 + 1.44085 w_k1_MAT_s017
                                 + 1.08013 w_k1_MSFT_s017
                                 + 1.13893 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.21844 w_k1_ADM_s017
                                 + 1.16927 w_k1_AXP_s017
                                 + 1.19632 w_k1_DIS_s017
                                 + 1.09012 w_k1_FDX_s017
                                 + 0.851662 w_k1_IBM_s017
                                 + 1.03186 w_k1_INTC_s017
                                 + 0.961311 w_k1_KO_s017
                                 + 0.965751 w_k1_MAT_s017
                                 + 1.0183 w_k1_MSFT_s017 + 1.04082 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09983 w_k1_ADM_s017
                                 + 0.987753 w_k1_AXP_s017
                                 + 1.15806 w_k1_DIS_s017
                                 + 1.00457 w_k1_FDX_s017
                                 + 1.08791 w_k1_IBM_s017
                                 + 1.04695 w_k1_INTC_s017
                                 + 1.18743 w_k1_KO_s017 + 1.17439 w_k1_MAT_s017
                                 + 1.00094 w_k1_MSFT_s017
                                 + 1.07839 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.27536 w_k1_ADM_s017
                                 + 1.04101 w_k1_AXP_s017
                                 + 0.903123 w_k1_DIS_s017
                                 + 0.904718 w_k1_FDX_s017
                                 + 1.11368 w_k1_IBM_s017
                                 + 0.79102 w_k1_INTC_s017
                                 + 1.10993 w_k1_KO_s017 + 1.52839 w_k1_MAT_s017
                                 + 0.811917 w_k1_MSFT_s017
                                 + 1.06591 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.28318 w_k1_ADM_s017
                                 + 1.61773 w_k1_AXP_s017 + 1.4119 w_k1_DIS_s017
                                 + 1.24396 w_k1_FDX_s017
                                 + 1.45651 w_k1_IBM_s017
                                 + 1.38931 w_k1_INTC_s017
                                 + 1.43367 w_k1_KO_s017 + 1.54924 w_k1_MAT_s017
                                 + 1.19845 w_k1_MSFT_s017
                                 + 1.09418 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.08103 w_k1_ADM_s100
                                 + 0.806755 w_k1_AXP_s100
                                 + 0.848214 w_k1_DIS_s100
                                 + 0.829898 w_k1_FDX_s100
                                 + 0.942993 w_k1_IBM_s100
                                 + 0.78046 w_k1_INTC_s100
                                 + 0.892884 w_k1_KO_s100
                                 + 0.705975 w_k1_MAT_s100
                                 + 0.886241 w_k1_MSFT_s100
                                 + 0.874335 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.3244 w_k1_ADM_s100
                                 + 0.967727 w_k1_AXP_s100
                                 + 1.03727 w_k1_DIS_s100
                                 + 1.15314 w_k1_FDX_s100
                                 + 0.77282 w_k1_IBM_s100
                                 + 0.900963 w_k1_INTC_s100
                                 + 0.8643 w_k1_KO_s100 + 0.569843 w_k1_MAT_s100
                                 + 1.04969 w_k1_MSFT_s100
                                 + 0.920227 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.26769 w_k1_ADM_s100
                                 + 1.27186 w_k1_AXP_s100
                                 + 1.42175 w_k1_DIS_s100
                                 + 1.39083 w_k1_FDX_s100
                                 + 0.931382 w_k1_IBM_s100
                                 + 1.21756 w_k1_INTC_s100
                                 + 0.974417 w_k1_KO_s100
                                 + 0.747457 w_k1_MAT_s100
                                 + 1.27953 w_k1_MSFT_s100
                                 + 1.01472 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.48521 w_k1_ADM_s100
                                 + 1.2465 w_k1_AXP_s100 + 1.53713 w_k1_DIS_s100
                                 + 1.31895 w_k1_FDX_s100
                                 + 1.19448 w_k1_IBM_s100
                                 + 1.10477 w_k1_INTC_s100
                                 + 1.20043 w_k1_KO_s100 + 1.42285 w_k1_MAT_s100
                                 + 1.37346 w_k1_MSFT_s100 + 1.0719 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.08792 w_k1_ADM_s100
                                 + 0.938226 w_k1_AXP_s100
                                 + 0.890087 w_k1_DIS_s100
                                 + 1.06043 w_k1_FDX_s100
                                 + 0.892856 w_k1_IBM_s100
                                 + 0.797738 w_k1_INTC_s100
                                 + 0.833258 w_k1_KO_s100
                                 + 0.824318 w_k1_MAT_s100
                                 + 0.966945 w_k1_MSFT_s100
                                 + 0.993758 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.20746 w_k1_ADM_s100
                                 + 1.10769 w_k1_AXP_s100
                                 + 1.29917 w_k1_DIS_s100 + 0.9949 w_k1_FDX_s100
                                 + 1.44624 w_k1_IBM_s100
                                 + 0.955425 w_k1_INTC_s100
                                 + 1.30057 w_k1_KO_s100 + 1.39281 w_k1_MAT_s100
                                 + 1.03711 w_k1_MSFT_s100
                                 + 1.12683 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.43775 w_k1_ADM_s100
                                 + 1.42738 w_k1_AXP_s100
                                 + 1.20163 w_k1_DIS_s100
                                 + 1.26481 w_k1_FDX_s100
                                 + 1.23739 w_k1_IBM_s100
                                 + 1.12628 w_k1_INTC_s100
                                 + 1.22237 w_k1_KO_s100 + 1.49393 w_k1_MAT_s100
                                 + 1.24887 w_k1_MSFT_s100
                                 + 1.05495 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.926201 w_k1_ADM_s100
                                 + 0.961037 w_k1_AXP_s100
                                 + 0.829572 w_k1_DIS_s100
                                 + 0.910394 w_k1_FDX_s100
                                 + 1.11683 w_k1_IBM_s100
                                 + 0.886884 w_k1_INTC_s100
                                 + 0.94532 w_k1_KO_s100 + 1.01695 w_k1_MAT_s100
                                 + 0.846929 w_k1_MSFT_s100
                                 + 0.887504 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.8817 w_k1_ADM_s101
                                 + 0.735881 w_k1_AXP_s101
                                 + 0.624185 w_k1_DIS_s101
                                 + 0.755859 w_k1_FDX_s101
                                 + 0.800512 w_k1_IBM_s101
                                 + 0.875215 w_k1_INTC_s101
                                 + 0.756737 w_k1_KO_s101
                                 + 0.602212 w_k1_MAT_s101
                                 + 0.704907 w_k1_MSFT_s101
                                 + 0.708705 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.11103 w_k1_ADM_s101
                                 + 1.587 w_k1_AXP_s101 + 1.55548 w_k1_DIS_s101
                                 + 1.40283 w_k1_FDX_s101
                                 + 1.22642 w_k1_IBM_s101
                                 + 1.33829 w_k1_INTC_s101
                                 + 1.14075 w_k1_KO_s101 + 1.41496 w_k1_MAT_s101
                                 + 1.14292 w_k1_MSFT_s101
                                 + 1.08957 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12366 w_k1_ADM_s101
                                 + 1.06692 w_k1_AXP_s101
                                 + 0.850171 w_k1_DIS_s101
                                 + 0.98638 w_k1_FDX_s101
                                 + 1.04977 w_k1_IBM_s101
                                 + 0.936333 w_k1_INTC_s101
                                 + 0.872278 w_k1_KO_s101
                                 + 0.93638 w_k1_MAT_s101
                                 + 0.83226 w_k1_MSFT_s101
                                 + 0.968194 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.22457 w_k1_ADM_s101
                                 + 1.04492 w_k1_AXP_s101
                                 + 1.35755 w_k1_DIS_s101
                                 + 1.22396 w_k1_FDX_s101
                                 + 1.08099 w_k1_IBM_s101
                                 + 1.13403 w_k1_INTC_s101
                                 + 1.18906 w_k1_KO_s101 + 1.15198 w_k1_MAT_s101
                                 + 1.05617 w_k1_MSFT_s101
                                 + 0.998079 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.98707 w_k1_ADM_s101
                                 + 0.785554 w_k1_AXP_s101
                                 + 0.956788 w_k1_DIS_s101
                                 + 1.00323 w_k1_FDX_s101
                                 + 0.787972 w_k1_IBM_s101
                                 + 1.03992 w_k1_INTC_s101
                                 + 0.850314 w_k1_KO_s101
                                 + 0.706367 w_k1_MAT_s101
                                 + 0.88629 w_k1_MSFT_s101
                                 + 0.941954 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.967452 w_k1_ADM_s101
                                 + 0.948773 w_k1_AXP_s101
                                 + 0.858683 w_k1_DIS_s101
                                 + 0.968543 w_k1_FDX_s101
                                 + 1.03083 w_k1_IBM_s101
                                 + 1.06766 w_k1_INTC_s101
                                 + 0.854787 w_k1_KO_s101
                                 + 0.952496 w_k1_MAT_s101
                                 + 0.870835 w_k1_MSFT_s101
                                 + 0.973739 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.946948 w_k1_ADM_s101
                                 + 1.1347 w_k1_AXP_s101 + 1.13123 w_k1_DIS_s101
                                 + 1.0554 w_k1_FDX_s101 + 1.29926 w_k1_IBM_s101
                                 + 0.955058 w_k1_INTC_s101
                                 + 1.19164 w_k1_KO_s101 + 1.36112 w_k1_MAT_s101
                                 + 0.994424 w_k1_MSFT_s101
                                 + 1.04879 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.698823 w_k1_ADM_s101
                                 + 0.83192 w_k1_AXP_s101
                                 + 0.768637 w_k1_DIS_s101
                                 + 0.817318 w_k1_FDX_s101 + 1.125 w_k1_IBM_s101
                                 + 0.771872 w_k1_INTC_s101
                                 + 0.835742 w_k1_KO_s101
                                 + 1.46525 w_k1_MAT_s101
                                 + 0.744525 w_k1_MSFT_s101
                                 + 0.830944 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.08509 w_k1_ADM_s102
                                 + 0.747002 w_k1_AXP_s102
                                 + 0.916864 w_k1_DIS_s102
                                 + 1.01601 w_k1_FDX_s102
                                 + 0.647943 w_k1_IBM_s102
                                 + 0.835539 w_k1_INTC_s102
                                 + 0.897623 w_k1_KO_s102
                                 + 0.687342 w_k1_MAT_s102
                                 + 0.923133 w_k1_MSFT_s102
                                 + 1.02768 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.92595 w_k1_ADM_s102
                                 + 0.938516 w_k1_AXP_s102
                                 + 1.13984 w_k1_DIS_s102
                                 + 1.01337 w_k1_FDX_s102
                                 + 1.04678 w_k1_IBM_s102
                                 + 1.03946 w_k1_INTC_s102 + 1.1122 w_k1_KO_s102
                                 + 0.952517 w_k1_MAT_s102
                                 + 1.01529 w_k1_MSFT_s102
                                 + 0.981044 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.959305 w_k1_ADM_s102
                                 + 0.813424 w_k1_AXP_s102
                                 + 0.920362 w_k1_DIS_s102
                                 + 0.883311 w_k1_FDX_s102
                                 + 1.31815 w_k1_IBM_s102
                                 + 0.960376 w_k1_INTC_s102
                                 + 1.09703 w_k1_KO_s102 + 1.03092 w_k1_MAT_s102
                                 + 0.819493 w_k1_MSFT_s102
                                 + 0.980422 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.995903 w_k1_ADM_s102
                                 + 1.04013 w_k1_AXP_s102
                                 + 0.877668 w_k1_DIS_s102
                                 + 0.952505 w_k1_FDX_s102
                                 + 0.937455 w_k1_IBM_s102
                                 + 0.918056 w_k1_INTC_s102
                                 + 0.880463 w_k1_KO_s102
                                 + 1.22283 w_k1_MAT_s102
                                 + 0.89508 w_k1_MSFT_s102
                                 + 0.96234 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.29611 w_k1_ADM_s102
                                 + 1.24561 w_k1_AXP_s102
                                 + 1.18256 w_k1_DIS_s102
                                 + 1.26946 w_k1_FDX_s102
                                 + 0.89827 w_k1_IBM_s102
                                 + 1.19723 w_k1_INTC_s102
                                 + 0.939558 w_k1_KO_s102
                                 + 0.809625 w_k1_MAT_s102
                                 + 1.08267 w_k1_MSFT_s102
                                 + 1.03549 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.63511 w_k1_ADM_s102
                                 + 0.696628 w_k1_AXP_s102
                                 + 0.682815 w_k1_DIS_s102
                                 + 0.802705 w_k1_FDX_s102
                                 + 0.702001 w_k1_IBM_s102
                                 + 0.88785 w_k1_INTC_s102
                                 + 0.664817 w_k1_KO_s102
                                 + 0.554514 w_k1_MAT_s102
                                 + 0.82895 w_k1_MSFT_s102
                                 + 0.794733 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.06414 w_k1_ADM_s102
                                 + 0.970793 w_k1_AXP_s102
                                 + 1.57507 w_k1_DIS_s102
                                 + 1.17779 w_k1_FDX_s102 + 1.1396 w_k1_IBM_s102
                                 + 0.96499 w_k1_INTC_s102
                                 + 1.29904 w_k1_KO_s102
                                 + 0.981889 w_k1_MAT_s102
                                 + 1.07573 w_k1_MSFT_s102
                                 + 1.21407 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.966972 w_k1_ADM_s102
                                 + 0.687945 w_k1_AXP_s102
                                 + 0.728728 w_k1_DIS_s102
                                 + 0.76572 w_k1_FDX_s102
                                 + 0.935109 w_k1_IBM_s102
                                 + 0.822906 w_k1_INTC_s102
                                 + 0.979987 w_k1_KO_s102
                                 + 0.914497 w_k1_MAT_s102
                                 + 0.699345 w_k1_MSFT_s102
                                 + 0.892604 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.42976 w_k1_ADM_s103
                                 + 1.63533 w_k1_AXP_s103
                                 + 1.21152 w_k1_DIS_s103
                                 + 1.33102 w_k1_FDX_s103
                                 + 1.24261 w_k1_IBM_s103
                                 + 1.12244 w_k1_INTC_s103 + 1.1766 w_k1_KO_s103
                                 + 1.26555 w_k1_MAT_s103
                                 + 1.23457 w_k1_MSFT_s103
                                 + 1.28113 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.831497 w_k1_ADM_s103
                                 + 0.889048 w_k1_AXP_s103
                                 + 0.675609 w_k1_DIS_s103
                                 + 0.744465 w_k1_FDX_s103
                                 + 1.1977 w_k1_IBM_s103
                                 + 0.795786 w_k1_INTC_s103
                                 + 0.995151 w_k1_KO_s103
                                 + 1.18402 w_k1_MAT_s103
                                 + 0.821153 w_k1_MSFT_s103
                                 + 0.859007 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05158 w_k1_ADM_s103
                                 + 1.05635 w_k1_AXP_s103
                                 + 0.981661 w_k1_DIS_s103
                                 + 1.0041 w_k1_FDX_s103
                                 + 0.908731 w_k1_IBM_s103
                                 + 0.980938 w_k1_INTC_s103
                                 + 0.920442 w_k1_KO_s103
                                 + 0.851731 w_k1_MAT_s103
                                 + 1.03525 w_k1_MSFT_s103
                                 + 0.886189 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.01179 w_k1_ADM_s103
                                 + 1.06492 w_k1_AXP_s103
                                 + 0.942087 w_k1_DIS_s103
                                 + 0.925513 w_k1_FDX_s103
                                 + 1.24402 w_k1_IBM_s103
                                 + 1.01667 w_k1_INTC_s103
                                 + 1.11761 w_k1_KO_s103
                                 + 0.897072 w_k1_MAT_s103
                                 + 1.0121 w_k1_MSFT_s103
                                 + 0.999906 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.74317 w_k1_ADM_s103
                                 + 0.750763 w_k1_AXP_s103
                                 + 0.588746 w_k1_DIS_s103
                                 + 0.777117 w_k1_FDX_s103
                                 + 0.777606 w_k1_IBM_s103
                                 + 0.822864 w_k1_INTC_s103
                                 + 0.646434 w_k1_KO_s103
                                 + 0.441546 w_k1_MAT_s103
                                 + 0.799259 w_k1_MSFT_s103
                                 + 0.740327 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.30196 w_k1_ADM_s103
                                 + 1.20501 w_k1_AXP_s103 + 1.3526 w_k1_DIS_s103
                                 + 1.32498 w_k1_FDX_s103
                                 + 0.854884 w_k1_IBM_s103
                                 + 1.31904 w_k1_INTC_s103 + 1.0736 w_k1_KO_s103
                                 + 0.937531 w_k1_MAT_s103
                                 + 1.37908 w_k1_MSFT_s103
                                 + 0.944748 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.989782 w_k1_ADM_s103
                                 + 1.18207 w_k1_AXP_s103
                                 + 1.56478 w_k1_DIS_s103
                                 + 1.19375 w_k1_FDX_s103
                                 + 1.27391 w_k1_IBM_s103
                                 + 1.18324 w_k1_INTC_s103
                                 + 1.15498 w_k1_KO_s103 + 1.01623 w_k1_MAT_s103
                                 + 1.23987 w_k1_MSFT_s103
                                 + 1.14768 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.28427 w_k1_ADM_s103
                                 + 1.40847 w_k1_AXP_s103
                                 + 1.20506 w_k1_DIS_s103
                                 + 1.15275 w_k1_FDX_s103
                                 + 1.57944 w_k1_IBM_s103
                                 + 1.04309 w_k1_INTC_s103 + 1.0836 w_k1_KO_s103
                                 + 1.29132 w_k1_MAT_s103
                                 + 1.07977 w_k1_MSFT_s103
                                 + 0.977437 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.760349 w_k1_ADM_s104
                                 + 0.631283 w_k1_AXP_s104
                                 + 0.586052 w_k1_DIS_s104
                                 + 0.709898 w_k1_FDX_s104
                                 + 0.954356 w_k1_IBM_s104
                                 + 0.703507 w_k1_INTC_s104
                                 + 0.784602 w_k1_KO_s104
                                 + 0.815038 w_k1_MAT_s104
                                 + 0.808578 w_k1_MSFT_s104
                                 + 0.789228 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13227 w_k1_ADM_s104
                                 + 1.13654 w_k1_AXP_s104
                                 + 1.02249 w_k1_DIS_s104
                                 + 1.09475 w_k1_FDX_s104
                                 + 1.09465 w_k1_IBM_s104
                                 + 0.936474 w_k1_INTC_s104
                                 + 0.931126 w_k1_KO_s104
                                 + 1.33362 w_k1_MAT_s104
                                 + 1.04222 w_k1_MSFT_s104 + 1.0504 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.21253 w_k1_ADM_s104
                                 + 1.03773 w_k1_AXP_s104
                                 + 0.97021 w_k1_DIS_s104
                                 + 0.906009 w_k1_FDX_s104
                                 + 1.44003 w_k1_IBM_s104
                                 + 0.932176 w_k1_INTC_s104
                                 + 0.982315 w_k1_KO_s104
                                 + 1.23022 w_k1_MAT_s104
                                 + 0.902673 w_k1_MSFT_s104
                                 + 0.930754 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.32306 w_k1_ADM_s104
                                 + 1.82078 w_k1_AXP_s104
                                 + 2.08364 w_k1_DIS_s104
                                 + 1.52993 w_k1_FDX_s104
                                 + 1.51891 w_k1_IBM_s104
                                 + 1.31684 w_k1_INTC_s104
                                 + 1.72358 w_k1_KO_s104 + 2.51797 w_k1_MAT_s104
                                 + 1.31636 w_k1_MSFT_s104
                                 + 1.31829 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.804156 w_k1_ADM_s104
                                 + 0.74215 w_k1_AXP_s104
                                 + 0.917998 w_k1_DIS_s104
                                 + 0.939264 w_k1_FDX_s104
                                 + 0.792006 w_k1_IBM_s104
                                 + 0.908739 w_k1_INTC_s104
                                 + 0.790519 w_k1_KO_s104
                                 + 0.772709 w_k1_MAT_s104
                                 + 1.02861 w_k1_MSFT_s104
                                 + 0.831489 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.16083 w_k1_ADM_s104
                                 + 0.853118 w_k1_AXP_s104
                                 + 1.31096 w_k1_DIS_s104
                                 + 1.07436 w_k1_FDX_s104
                                 + 1.07924 w_k1_IBM_s104
                                 + 1.20968 w_k1_INTC_s104 + 1.0407 w_k1_KO_s104
                                 + 0.797816 w_k1_MAT_s104
                                 + 1.16668 w_k1_MSFT_s104
                                 + 1.06539 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.54543 w_k1_ADM_s104
                                 + 1.29917 w_k1_AXP_s104
                                 + 1.56287 w_k1_DIS_s104
                                 + 1.36358 w_k1_FDX_s104
                                 + 0.984576 w_k1_IBM_s104
                                 + 1.27568 w_k1_INTC_s104
                                 + 1.11726 w_k1_KO_s104
                                 + 0.696852 w_k1_MAT_s104
                                 + 1.28959 w_k1_MSFT_s104
                                 + 1.21493 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.932887 w_k1_ADM_s104
                                 + 0.790801 w_k1_AXP_s104
                                 + 0.608177 w_k1_DIS_s104
                                 + 0.833483 w_k1_FDX_s104
                                 + 0.759637 w_k1_IBM_s104
                                 + 0.855554 w_k1_INTC_s104
                                 + 0.684574 w_k1_KO_s104
                                 + 0.755065 w_k1_MAT_s104
                                 + 0.801414 w_k1_MSFT_s104
                                 + 0.750978 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.10189 w_k1_ADM_s105
                                 + 1.02902 w_k1_AXP_s105
                                 + 1.29609 w_k1_DIS_s105
                                 + 1.10703 w_k1_FDX_s105
                                 + 1.19638 w_k1_IBM_s105
                                 + 1.03984 w_k1_INTC_s105
                                 + 1.11158 w_k1_KO_s105 + 1.03804 w_k1_MAT_s105
                                 + 1.30716 w_k1_MSFT_s105
                                 + 1.02634 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.971379 w_k1_ADM_s105
                                 + 0.777561 w_k1_AXP_s105
                                 + 0.786733 w_k1_DIS_s105
                                 + 0.797139 w_k1_FDX_s105
                                 + 0.920944 w_k1_IBM_s105
                                 + 0.758482 w_k1_INTC_s105
                                 + 0.744444 w_k1_KO_s105
                                 + 0.817758 w_k1_MAT_s105
                                 + 0.959742 w_k1_MSFT_s105
                                 + 0.898429 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.04568 w_k1_ADM_s105
                                 + 1.47846 w_k1_AXP_s105
                                 + 1.02435 w_k1_DIS_s105
                                 + 1.02486 w_k1_FDX_s105
                                 + 1.36768 w_k1_IBM_s105
                                 + 0.791258 w_k1_INTC_s105
                                 + 0.942862 w_k1_KO_s105
                                 + 1.41149 w_k1_MAT_s105
                                 + 0.967917 w_k1_MSFT_s105
                                 + 1.0197 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.873844 w_k1_ADM_s105
                                 + 0.646393 w_k1_AXP_s105
                                 + 0.501598 w_k1_DIS_s105
                                 + 0.653945 w_k1_FDX_s105
                                 + 0.769914 w_k1_IBM_s105
                                 + 0.7401 w_k1_INTC_s105
                                 + 0.666614 w_k1_KO_s105
                                 + 0.708969 w_k1_MAT_s105
                                 + 0.763625 w_k1_MSFT_s105
                                 + 0.83794 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.11409 w_k1_ADM_s105
                                 + 1.03774 w_k1_AXP_s105
                                 + 0.757183 w_k1_DIS_s105
                                 + 1.02955 w_k1_FDX_s105
                                 + 0.755375 w_k1_IBM_s105
                                 + 0.651184 w_k1_INTC_s105
                                 + 0.718738 w_k1_KO_s105 + 1.0404 w_k1_MAT_s105
                                 + 0.915319 w_k1_MSFT_s105
                                 + 0.92083 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.19213 w_k1_ADM_s105
                                 + 1.39449 w_k1_AXP_s105
                                 + 1.21985 w_k1_DIS_s105
                                 + 1.26922 w_k1_FDX_s105
                                 + 0.871078 w_k1_IBM_s105
                                 + 1.08841 w_k1_INTC_s105
                                 + 1.06634 w_k1_KO_s105
                                 + 0.892356 w_k1_MAT_s105
                                 + 1.3403 w_k1_MSFT_s105 + 1.16138 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.88348 w_k1_ADM_s105
                                 + 1.63596 w_k1_AXP_s105
                                 + 1.98577 w_k1_DIS_s105
                                 + 1.47783 w_k1_FDX_s105
                                 + 1.41389 w_k1_IBM_s105
                                 + 1.14285 w_k1_INTC_s105
                                 + 1.46496 w_k1_KO_s105 + 1.40769 w_k1_MAT_s105
                                 + 1.60277 w_k1_MSFT_s105
                                 + 1.34686 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.03316 w_k1_ADM_s105
                                 + 0.891347 w_k1_AXP_s105
                                 + 1.0965 w_k1_DIS_s105 + 1.0393 w_k1_FDX_s105
                                 + 0.772763 w_k1_IBM_s105
                                 + 1.09663 w_k1_INTC_s105
                                 + 0.989008 w_k1_KO_s105
                                 + 0.646192 w_k1_MAT_s105
                                 + 1.23713 w_k1_MSFT_s105
                                 + 1.01629 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.913488 w_k1_ADM_s106
                                 + 0.777519 w_k1_AXP_s106
                                 + 0.704505 w_k1_DIS_s106
                                 + 0.789006 w_k1_FDX_s106
                                 + 0.637851 w_k1_IBM_s106
                                 + 0.935052 w_k1_INTC_s106
                                 + 0.729685 w_k1_KO_s106
                                 + 0.516386 w_k1_MAT_s106
                                 + 0.882765 w_k1_MSFT_s106
                                 + 0.79114 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.784716 w_k1_ADM_s106
                                 + 0.834128 w_k1_AXP_s106
                                 + 0.742609 w_k1_DIS_s106
                                 + 0.743813 w_k1_FDX_s106
                                 + 1.02595 w_k1_IBM_s106
                                 + 0.774496 w_k1_INTC_s106
                                 + 0.773828 w_k1_KO_s106
                                 + 0.953059 w_k1_MAT_s106
                                 + 0.815319 w_k1_MSFT_s106
                                 + 0.932453 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.984764 w_k1_ADM_s106
                                 + 0.962457 w_k1_AXP_s106
                                 + 0.966622 w_k1_DIS_s106
                                 + 1.04502 w_k1_FDX_s106
                                 + 0.708727 w_k1_IBM_s106
                                 + 1.12191 w_k1_INTC_s106
                                 + 0.843116 w_k1_KO_s106
                                 + 0.779255 w_k1_MAT_s106
                                 + 1.07077 w_k1_MSFT_s106
                                 + 0.928787 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.916693 w_k1_ADM_s106
                                 + 1.20698 w_k1_AXP_s106
                                 + 1.12567 w_k1_DIS_s106
                                 + 1.03153 w_k1_FDX_s106
                                 + 1.52184 w_k1_IBM_s106
                                 + 1.22493 w_k1_INTC_s106
                                 + 1.19746 w_k1_KO_s106 + 1.49568 w_k1_MAT_s106
                                 + 1.09458 w_k1_MSFT_s106 + 1.0399 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.09064 w_k1_ADM_s106
                                 + 1.13064 w_k1_AXP_s106
                                 + 1.08489 w_k1_DIS_s106
                                 + 1.02688 w_k1_FDX_s106
                                 + 1.07744 w_k1_IBM_s106
                                 + 1.06365 w_k1_INTC_s106
                                 + 0.87231 w_k1_KO_s106 + 1.0642 w_k1_MAT_s106
                                 + 1.03225 w_k1_MSFT_s106
                                 + 0.948934 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.03221 w_k1_ADM_s106
                                 + 0.929729 w_k1_AXP_s106
                                 + 1.11675 w_k1_DIS_s106
                                 + 1.08933 w_k1_FDX_s106
                                 + 0.899651 w_k1_IBM_s106
                                 + 1.0609 w_k1_INTC_s106
                                 + 0.944208 w_k1_KO_s106
                                 + 0.652856 w_k1_MAT_s106
                                 + 1.18051 w_k1_MSFT_s106
                                 + 0.980289 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.41509 w_k1_ADM_s106
                                 + 1.50936 w_k1_AXP_s106
                                 + 1.00537 w_k1_DIS_s106
                                 + 1.27739 w_k1_FDX_s106
                                 + 0.957714 w_k1_IBM_s106
                                 + 1.13767 w_k1_INTC_s106
                                 + 0.875338 w_k1_KO_s106
                                 + 1.14875 w_k1_MAT_s106
                                 + 1.14849 w_k1_MSFT_s106
                                 + 1.05031 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.34049 w_k1_ADM_s106
                                 + 1.42151 w_k1_AXP_s106
                                 + 1.79189 w_k1_DIS_s106
                                 + 1.35812 w_k1_FDX_s106
                                 + 1.16198 w_k1_IBM_s106
                                 + 1.14657 w_k1_INTC_s106
                                 + 1.30662 w_k1_KO_s106 + 1.76858 w_k1_MAT_s106
                                 + 1.40617 w_k1_MSFT_s106
                                 + 1.32071 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.19153 w_k1_ADM_s107
                                 + 1.34886 w_k1_AXP_s107
                                 + 1.27778 w_k1_DIS_s107
                                 + 1.08416 w_k1_FDX_s107
                                 + 1.37883 w_k1_IBM_s107
                                 + 1.28993 w_k1_INTC_s107
                                 + 1.23669 w_k1_KO_s107 + 1.04654 w_k1_MAT_s107
                                 + 1.14354 w_k1_MSFT_s107 + 1.2764 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.844493 w_k1_ADM_s107
                                 + 0.805252 w_k1_AXP_s107
                                 + 0.659186 w_k1_DIS_s107
                                 + 0.764453 w_k1_FDX_s107
                                 + 0.767363 w_k1_IBM_s107
                                 + 0.988145 w_k1_INTC_s107
                                 + 0.780537 w_k1_KO_s107
                                 + 0.643783 w_k1_MAT_s107
                                 + 0.847515 w_k1_MSFT_s107
                                 + 0.87227 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.875971 w_k1_ADM_s107
                                 + 0.898648 w_k1_AXP_s107
                                 + 0.797008 w_k1_DIS_s107
                                 + 0.947781 w_k1_FDX_s107
                                 + 0.827517 w_k1_IBM_s107
                                 + 0.971641 w_k1_INTC_s107
                                 + 0.747631 w_k1_KO_s107
                                 + 0.86221 w_k1_MAT_s107
                                 + 0.934578 w_k1_MSFT_s107
                                 + 0.980342 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.5041 w_k1_ADM_s107
                                 + 1.52857 w_k1_AXP_s107
                                 + 1.51281 w_k1_DIS_s107
                                 + 1.38784 w_k1_FDX_s107
                                 + 1.02249 w_k1_IBM_s107
                                 + 1.30424 w_k1_INTC_s107
                                 + 1.10476 w_k1_KO_s107
                                 + 0.787338 w_k1_MAT_s107
                                 + 1.32512 w_k1_MSFT_s107
                                 + 1.28548 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.688289 w_k1_ADM_s107
                                 + 0.853593 w_k1_AXP_s107
                                 + 0.713552 w_k1_DIS_s107
                                 + 0.704134 w_k1_FDX_s107
                                 + 1.32706 w_k1_IBM_s107
                                 + 1.02719 w_k1_INTC_s107
                                 + 0.887218 w_k1_KO_s107
                                 + 1.09342 w_k1_MAT_s107
                                 + 0.893965 w_k1_MSFT_s107
                                 + 0.84647 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.01913 w_k1_ADM_s107
                                 + 0.960085 w_k1_AXP_s107
                                 + 1.24311 w_k1_DIS_s107
                                 + 1.02091 w_k1_FDX_s107
                                 + 0.906894 w_k1_IBM_s107
                                 + 1.25527 w_k1_INTC_s107
                                 + 1.05634 w_k1_KO_s107
                                 + 0.803653 w_k1_MAT_s107
                                 + 1.19662 w_k1_MSFT_s107
                                 + 1.12986 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.983549 w_k1_ADM_s107
                                 + 1.15923 w_k1_AXP_s107
                                 + 1.01067 w_k1_DIS_s107
                                 + 0.925913 w_k1_FDX_s107
                                 + 1.45443 w_k1_IBM_s107
                                 + 1.32472 w_k1_INTC_s107
                                 + 1.20533 w_k1_KO_s107 + 1.49505 w_k1_MAT_s107
                                 + 1.03475 w_k1_MSFT_s107
                                 + 1.19282 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.22321 w_k1_ADM_s107
                                 + 1.0457 w_k1_AXP_s107 + 1.12685 w_k1_DIS_s107
                                 + 1.23393 w_k1_FDX_s107
                                 + 0.759862 w_k1_IBM_s107
                                 + 1.30377 w_k1_INTC_s107
                                 + 0.868367 w_k1_KO_s107
                                 + 0.432677 w_k1_MAT_s107
                                 + 1.16475 w_k1_MSFT_s107
                                 + 1.01636 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.10233 w_k1_ADM_s110
                                 + 0.677023 w_k1_AXP_s110
                                 + 0.660612 w_k1_DIS_s110
                                 + 0.731506 w_k1_FDX_s110
                                 + 0.784904 w_k1_IBM_s110
                                 + 0.734598 w_k1_INTC_s110
                                 + 0.744126 w_k1_KO_s110
                                 + 0.709522 w_k1_MAT_s110
                                 + 0.882399 w_k1_MSFT_s110
                                 + 0.859298 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.59776 w_k1_ADM_s110
                                 + 1.58424 w_k1_AXP_s110
                                 + 1.57303 w_k1_DIS_s110
                                 + 1.18259 w_k1_FDX_s110
                                 + 1.48268 w_k1_IBM_s110
                                 + 1.06221 w_k1_INTC_s110
                                 + 1.34924 w_k1_KO_s110 + 1.79066 w_k1_MAT_s110
                                 + 1.28961 w_k1_MSFT_s110
                                 + 1.27308 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.35681 w_k1_ADM_s110
                                 + 0.871735 w_k1_AXP_s110
                                 + 1.03341 w_k1_DIS_s110
                                 + 1.00131 w_k1_FDX_s110
                                 + 0.908854 w_k1_IBM_s110
                                 + 1.1111 w_k1_INTC_s110 + 1.0705 w_k1_KO_s110
                                 + 0.837979 w_k1_MAT_s110
                                 + 1.25881 w_k1_MSFT_s110
                                 + 1.04806 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.952193 w_k1_ADM_s110
                                 + 0.674268 w_k1_AXP_s110
                                 + 0.703453 w_k1_DIS_s110
                                 + 0.835978 w_k1_FDX_s110
                                 + 0.593565 w_k1_IBM_s110
                                 + 0.868737 w_k1_INTC_s110
                                 + 0.660828 w_k1_KO_s110
                                 + 0.512416 w_k1_MAT_s110
                                 + 0.912774 w_k1_MSFT_s110
                                 + 0.968517 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.921504 w_k1_ADM_s110
                                 + 0.680305 w_k1_AXP_s110
                                 + 0.781238 w_k1_DIS_s110
                                 + 0.742647 w_k1_FDX_s110
                                 + 1.23918 w_k1_IBM_s110
                                 + 0.779442 w_k1_INTC_s110
                                 + 0.914696 w_k1_KO_s110
                                 + 1.06603 w_k1_MAT_s110
                                 + 0.82382 w_k1_MSFT_s110
                                 + 0.916401 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.1699 w_k1_ADM_s110
                                 + 1.09518 w_k1_AXP_s110
                                 + 1.11637 w_k1_DIS_s110
                                 + 1.17682 w_k1_FDX_s110
                                 + 0.842576 w_k1_IBM_s110
                                 + 0.95557 w_k1_INTC_s110
                                 + 0.789646 w_k1_KO_s110
                                 + 0.92821 w_k1_MAT_s110
                                 + 0.988723 w_k1_MSFT_s110
                                 + 1.11308 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.3845 w_k1_ADM_s110
                                 + 1.24975 w_k1_AXP_s110
                                 + 1.40831 w_k1_DIS_s110
                                 + 1.28955 w_k1_FDX_s110
                                 + 0.88288 w_k1_IBM_s110
                                 + 1.29944 w_k1_INTC_s110
                                 + 0.902098 w_k1_KO_s110 + 0.9115 w_k1_MAT_s110
                                 + 1.34018 w_k1_MSFT_s110 + 1.2078 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.898954 w_k1_ADM_s110
                                 + 1.00319 w_k1_AXP_s110
                                 + 0.891136 w_k1_DIS_s110
                                 + 0.936742 w_k1_FDX_s110
                                 + 0.941961 w_k1_IBM_s110
                                 + 1.01978 w_k1_INTC_s110
                                 + 0.897529 w_k1_KO_s110
                                 + 0.92082 w_k1_MAT_s110
                                 + 1.00328 w_k1_MSFT_s110
                                 + 0.932388 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.00104 w_k1_ADM_s111
                                 + 1.33114 w_k1_AXP_s111
                                 + 1.40274 w_k1_DIS_s111
                                 + 1.20706 w_k1_FDX_s111
                                 + 0.952455 w_k1_IBM_s111
                                 + 1.31947 w_k1_INTC_s111
                                 + 1.20774 w_k1_KO_s111 + 1.35879 w_k1_MAT_s111
                                 + 1.37602 w_k1_MSFT_s111 + 1.1688 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.93384 w_k1_ADM_s111
                                 + 1.05669 w_k1_AXP_s111
                                 + 0.946773 w_k1_DIS_s111
                                 + 1.01412 w_k1_FDX_s111
                                 + 0.900623 w_k1_IBM_s111
                                 + 0.959341 w_k1_INTC_s111
                                 + 0.842219 w_k1_KO_s111
                                 + 0.914962 w_k1_MAT_s111
                                 + 0.955483 w_k1_MSFT_s111
                                 + 0.987635 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.659221 w_k1_ADM_s111
                                 + 0.647735 w_k1_AXP_s111
                                 + 0.719802 w_k1_DIS_s111
                                 + 0.745073 w_k1_FDX_s111
                                 + 0.9674 w_k1_IBM_s111
                                 + 0.85533 w_k1_INTC_s111
                                 + 0.830547 w_k1_KO_s111
                                 + 0.885699 w_k1_MAT_s111
                                 + 0.861692 w_k1_MSFT_s111
                                 + 0.748392 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.804583 w_k1_ADM_s111
                                 + 0.751537 w_k1_AXP_s111
                                 + 0.825413 w_k1_DIS_s111
                                 + 0.830096 w_k1_FDX_s111
                                 + 0.77323 w_k1_IBM_s111
                                 + 0.854214 w_k1_INTC_s111
                                 + 0.750825 w_k1_KO_s111
                                 + 0.808295 w_k1_MAT_s111
                                 + 0.894258 w_k1_MSFT_s111
                                 + 0.771388 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.733073 w_k1_ADM_s111
                                 + 1.11683 w_k1_AXP_s111 + 1.0153 w_k1_DIS_s111
                                 + 0.905755 w_k1_FDX_s111
                                 + 1.20988 w_k1_IBM_s111
                                 + 1.06365 w_k1_INTC_s111
                                 + 1.03208 w_k1_KO_s111 + 1.28803 w_k1_MAT_s111
                                 + 0.869736 w_k1_MSFT_s111
                                 + 0.900203 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.988924 w_k1_ADM_s111
                                 + 1.64116 w_k1_AXP_s111
                                 + 1.67632 w_k1_DIS_s111
                                 + 1.34171 w_k1_FDX_s111 + 1.1974 w_k1_IBM_s111
                                 + 1.49333 w_k1_INTC_s111
                                 + 1.14903 w_k1_KO_s111 + 1.44865 w_k1_MAT_s111
                                 + 1.2668 w_k1_MSFT_s111 + 1.19962 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.943582 w_k1_ADM_s111
                                 + 1.08592 w_k1_AXP_s111
                                 + 1.01343 w_k1_DIS_s111
                                 + 1.06045 w_k1_FDX_s111
                                 + 0.635621 w_k1_IBM_s111
                                 + 0.974035 w_k1_INTC_s111
                                 + 0.918344 w_k1_KO_s111
                                 + 0.779072 w_k1_MAT_s111
                                 + 1.10601 w_k1_MSFT_s111
                                 + 0.981267 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.842924 w_k1_ADM_s111
                                 + 1.10713 w_k1_AXP_s111
                                 + 1.20434 w_k1_DIS_s111
                                 + 1.03411 w_k1_FDX_s111
                                 + 1.50724 w_k1_IBM_s111
                                 + 1.09046 w_k1_INTC_s111
                                 + 1.40561 w_k1_KO_s111 + 1.77332 w_k1_MAT_s111
                                 + 1.06202 w_k1_MSFT_s111
                                 + 1.15656 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.998817 w_k1_ADM_s112
                                 + 0.842882 w_k1_AXP_s112
                                 + 0.983243 w_k1_DIS_s112
                                 + 0.991571 w_k1_FDX_s112
                                 + 0.841887 w_k1_IBM_s112
                                 + 1.07918 w_k1_INTC_s112
                                 + 0.900941 w_k1_KO_s112
                                 + 0.876194 w_k1_MAT_s112
                                 + 1.02257 w_k1_MSFT_s112
                                 + 1.09224 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.27461 w_k1_ADM_s112
                                 + 1.2491 w_k1_AXP_s112
                                 + 0.943975 w_k1_DIS_s112
                                 + 0.989954 w_k1_FDX_s112
                                 + 1.0374 w_k1_IBM_s112
                                 + 0.884391 w_k1_INTC_s112
                                 + 0.800514 w_k1_KO_s112
                                 + 0.99866 w_k1_MAT_s112
                                 + 0.911417 w_k1_MSFT_s112
                                 + 1.07214 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.09139 w_k1_ADM_s112
                                 + 1.17566 w_k1_AXP_s112
                                 + 1.32723 w_k1_DIS_s112
                                 + 1.16627 w_k1_FDX_s112
                                 + 0.86244 w_k1_IBM_s112
                                 + 1.07505 w_k1_INTC_s112
                                 + 0.988435 w_k1_KO_s112
                                 + 1.02358 w_k1_MAT_s112
                                 + 1.24278 w_k1_MSFT_s112
                                 + 1.14891 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.895931 w_k1_ADM_s112
                                 + 0.543651 w_k1_AXP_s112
                                 + 0.729094 w_k1_DIS_s112
                                 + 0.731719 w_k1_FDX_s112
                                 + 0.681998 w_k1_IBM_s112
                                 + 0.850122 w_k1_INTC_s112
                                 + 0.742406 w_k1_KO_s112
                                 + 0.388221 w_k1_MAT_s112
                                 + 0.759952 w_k1_MSFT_s112
                                 + 0.785184 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.900617 w_k1_ADM_s112
                                 + 0.747229 w_k1_AXP_s112
                                 + 0.797971 w_k1_DIS_s112
                                 + 0.753609 w_k1_FDX_s112
                                 + 1.15286 w_k1_IBM_s112
                                 + 0.963438 w_k1_INTC_s112
                                 + 1.06787 w_k1_KO_s112
                                 + 0.927236 w_k1_MAT_s112
                                 + 0.846702 w_k1_MSFT_s112
                                 + 0.968231 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.850047 w_k1_ADM_s112
                                 + 0.921804 w_k1_AXP_s112
                                 + 0.750586 w_k1_DIS_s112
                                 + 0.900654 w_k1_FDX_s112
                                 + 0.819304 w_k1_IBM_s112
                                 + 0.885868 w_k1_INTC_s112
                                 + 0.696717 w_k1_KO_s112
                                 + 0.678734 w_k1_MAT_s112
                                 + 0.778203 w_k1_MSFT_s112
                                 + 0.973399 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.17477 w_k1_ADM_s112
                                 + 1.34926 w_k1_AXP_s112
                                 + 1.40573 w_k1_DIS_s112 + 1.105 w_k1_FDX_s112
                                 + 1.35536 w_k1_IBM_s112
                                 + 1.07429 w_k1_INTC_s112
                                 + 1.15968 w_k1_KO_s112 + 1.22519 w_k1_MAT_s112
                                 + 1.04968 w_k1_MSFT_s112
                                 + 1.28163 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.57084 w_k1_ADM_s112
                                 + 1.44852 w_k1_AXP_s112
                                 + 1.73358 w_k1_DIS_s112
                                 + 1.42313 w_k1_FDX_s112
                                 + 1.31444 w_k1_IBM_s112
                                 + 1.33706 w_k1_INTC_s112
                                 + 1.47071 w_k1_KO_s112 + 1.43766 w_k1_MAT_s112
                                 + 1.30851 w_k1_MSFT_s112
                                 + 1.41717 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.03016 w_k1_ADM_s113
                                 + 1.10521 w_k1_AXP_s113
                                 + 0.965487 w_k1_DIS_s113
                                 + 0.985681 w_k1_FDX_s113
                                 + 1.05535 w_k1_IBM_s113
                                 + 0.940179 w_k1_INTC_s113
                                 + 1.03996 w_k1_KO_s113 + 1.08006 w_k1_MAT_s113
                                 + 0.832891 w_k1_MSFT_s113
                                 + 1.00351 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.16675 w_k1_ADM_s113
                                 + 1.1957 w_k1_AXP_s113 + 1.20392 w_k1_DIS_s113
                                 + 1.06565 w_k1_FDX_s113
                                 + 1.22915 w_k1_IBM_s113
                                 + 0.9406 w_k1_INTC_s113 + 1.06858 w_k1_KO_s113
                                 + 1.46606 w_k1_MAT_s113
                                 + 1.0414 w_k1_MSFT_s113 + 1.16716 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.986678 w_k1_ADM_s113
                                 + 0.858168 w_k1_AXP_s113
                                 + 1.25192 w_k1_DIS_s113
                                 + 1.13521 w_k1_FDX_s113
                                 + 0.924872 w_k1_IBM_s113
                                 + 1.24657 w_k1_INTC_s113 + 1.1657 w_k1_KO_s113
                                 + 0.836979 w_k1_MAT_s113
                                 + 1.22527 w_k1_MSFT_s113
                                 + 1.11549 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.97959 w_k1_ADM_s113
                                 + 1.14688 w_k1_AXP_s113
                                 + 1.43961 w_k1_DIS_s113
                                 + 1.15957 w_k1_FDX_s113
                                 + 1.55045 w_k1_IBM_s113
                                 + 1.12615 w_k1_INTC_s113
                                 + 1.34661 w_k1_KO_s113 + 1.59828 w_k1_MAT_s113
                                 + 1.20603 w_k1_MSFT_s113 + 1.1713 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.683181 w_k1_ADM_s113
                                 + 0.729314 w_k1_AXP_s113
                                 + 0.814387 w_k1_DIS_s113
                                 + 0.731661 w_k1_FDX_s113
                                 + 1.12205 w_k1_IBM_s113
                                 + 0.868211 w_k1_INTC_s113
                                 + 1.06498 w_k1_KO_s113 + 1.19243 w_k1_MAT_s113
                                 + 0.792612 w_k1_MSFT_s113
                                 + 0.891253 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.880483 w_k1_ADM_s113
                                 + 0.981715 w_k1_AXP_s113
                                 + 0.612794 w_k1_DIS_s113
                                 + 0.786396 w_k1_FDX_s113
                                 + 0.786606 w_k1_IBM_s113
                                 + 0.932733 w_k1_INTC_s113
                                 + 0.86519 w_k1_KO_s113
                                 + 0.910274 w_k1_MAT_s113
                                 + 0.891283 w_k1_MSFT_s113
                                 + 0.878291 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.15 w_k1_ADM_s113
                                 + 1.34964 w_k1_AXP_s113
                                 + 0.981972 w_k1_DIS_s113
                                 + 1.08033 w_k1_FDX_s113
                                 + 0.890008 w_k1_IBM_s113
                                 + 0.918274 w_k1_INTC_s113
                                 + 0.898184 w_k1_KO_s113
                                 + 0.951195 w_k1_MAT_s113
                                 + 0.910778 w_k1_MSFT_s113
                                 + 0.916888 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.857875 w_k1_ADM_s113
                                 + 0.700504 w_k1_AXP_s113
                                 + 0.800227 w_k1_DIS_s113
                                 + 0.897015 w_k1_FDX_s113
                                 + 0.708932 w_k1_IBM_s113
                                 + 0.723381 w_k1_INTC_s113
                                 + 0.791494 w_k1_KO_s113
                                 + 0.761875 w_k1_MAT_s113
                                 + 0.891189 w_k1_MSFT_s113
                                 + 0.863889 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.692777 w_k1_ADM_s114
                                 + 0.823053 w_k1_AXP_s114
                                 + 0.748085 w_k1_DIS_s114
                                 + 0.751463 w_k1_FDX_s114
                                 + 1.13536 w_k1_IBM_s114
                                 + 0.893196 w_k1_INTC_s114
                                 + 1.0393 w_k1_KO_s114 + 0.954633 w_k1_MAT_s114
                                 + 0.913333 w_k1_MSFT_s114
                                 + 0.935543 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.898315 w_k1_ADM_s114
                                 + 1.14422 w_k1_AXP_s114
                                 + 1.29728 w_k1_DIS_s114
                                 + 1.20472 w_k1_FDX_s114
                                 + 1.12639 w_k1_IBM_s114
                                 + 1.20938 w_k1_INTC_s114
                                 + 1.23536 w_k1_KO_s114
                                 + 0.979241 w_k1_MAT_s114
                                 + 1.15538 w_k1_MSFT_s114 + 1.1256 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.1835 w_k1_ADM_s114
                                 + 1.2788 w_k1_AXP_s114 + 1.20768 w_k1_DIS_s114
                                 + 1.21118 w_k1_FDX_s114
                                 + 0.857082 w_k1_IBM_s114
                                 + 1.11426 w_k1_INTC_s114
                                 + 0.987939 w_k1_KO_s114
                                 + 0.805864 w_k1_MAT_s114
                                 + 1.23519 w_k1_MSFT_s114
                                 + 1.04802 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.879049 w_k1_ADM_s114
                                 + 0.865359 w_k1_AXP_s114
                                 + 0.736413 w_k1_DIS_s114
                                 + 0.844471 w_k1_FDX_s114
                                 + 0.801303 w_k1_IBM_s114
                                 + 0.900788 w_k1_INTC_s114
                                 + 0.870485 w_k1_KO_s114
                                 + 0.622309 w_k1_MAT_s114
                                 + 0.884202 w_k1_MSFT_s114
                                 + 0.89074 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.918122 w_k1_ADM_s114
                                 + 1.03944 w_k1_AXP_s114
                                 + 0.980038 w_k1_DIS_s114
                                 + 1.0551 w_k1_FDX_s114
                                 + 0.932375 w_k1_IBM_s114
                                 + 1.17343 w_k1_INTC_s114
                                 + 0.984675 w_k1_KO_s114
                                 + 0.693842 w_k1_MAT_s114
                                 + 1.04696 w_k1_MSFT_s114
                                 + 0.930589 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.892815 w_k1_ADM_s114
                                 + 1.36209 w_k1_AXP_s114
                                 + 0.99037 w_k1_DIS_s114
                                 + 1.03402 w_k1_FDX_s114
                                 + 1.10469 w_k1_IBM_s114
                                 + 0.956168 w_k1_INTC_s114
                                 + 0.999006 w_k1_KO_s114 + 1.1231 w_k1_MAT_s114
                                 + 1.04467 w_k1_MSFT_s114
                                 + 1.11345 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.920521 w_k1_ADM_s114
                                 + 1.26901 w_k1_AXP_s114
                                 + 1.41249 w_k1_DIS_s114
                                 + 1.10633 w_k1_FDX_s114
                                 + 1.53316 w_k1_IBM_s114
                                 + 0.968107 w_k1_INTC_s114
                                 + 1.4355 w_k1_KO_s114 + 1.60772 w_k1_MAT_s114
                                 + 1.17326 w_k1_MSFT_s114
                                 + 1.17287 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.626424 w_k1_ADM_s114
                                 + 0.562335 w_k1_AXP_s114
                                 + 0.544738 w_k1_DIS_s114
                                 + 0.626712 w_k1_FDX_s114
                                 + 0.76437 w_k1_IBM_s114
                                 + 0.623783 w_k1_INTC_s114
                                 + 0.734558 w_k1_KO_s114
                                 + 0.671189 w_k1_MAT_s114
                                 + 0.72846 w_k1_MSFT_s114
                                 + 0.81698 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.0744 w_k1_ADM_s115
                                 + 1.37518 w_k1_AXP_s115
                                 + 1.35618 w_k1_DIS_s115
                                 + 1.06075 w_k1_FDX_s115 + 1.7203 w_k1_IBM_s115
                                 + 1.11441 w_k1_INTC_s115
                                 + 1.63297 w_k1_KO_s115 + 1.71263 w_k1_MAT_s115
                                 + 1.04247 w_k1_MSFT_s115
                                 + 1.17308 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.22681 w_k1_ADM_s115
                                 + 1.55639 w_k1_AXP_s115
                                 + 0.868447 w_k1_DIS_s115
                                 + 1.14577 w_k1_FDX_s115
                                 + 0.821098 w_k1_IBM_s115
                                 + 0.908725 w_k1_INTC_s115
                                 + 0.94299 w_k1_KO_s115
                                 + 0.954525 w_k1_MAT_s115
                                 + 0.958974 w_k1_MSFT_s115
                                 + 0.978123 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06231 w_k1_ADM_s115
                                 + 1.18147 w_k1_AXP_s115
                                 + 1.34667 w_k1_DIS_s115
                                 + 1.26845 w_k1_FDX_s115
                                 + 0.828584 w_k1_IBM_s115
                                 + 1.42084 w_k1_INTC_s115
                                 + 1.00521 w_k1_KO_s115
                                 + 0.603588 w_k1_MAT_s115
                                 + 1.11657 w_k1_MSFT_s115
                                 + 1.08098 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.941507 w_k1_ADM_s115
                                 + 0.769117 w_k1_AXP_s115
                                 + 0.928605 w_k1_DIS_s115
                                 + 0.950861 w_k1_FDX_s115
                                 + 0.757044 w_k1_IBM_s115
                                 + 0.788479 w_k1_INTC_s115
                                 + 0.854108 w_k1_KO_s115
                                 + 0.523957 w_k1_MAT_s115
                                 + 0.896853 w_k1_MSFT_s115
                                 + 0.947132 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.985755 w_k1_ADM_s115
                                 + 0.966108 w_k1_AXP_s115
                                 + 0.870431 w_k1_DIS_s115
                                 + 0.881542 w_k1_FDX_s115
                                 + 1.32253 w_k1_IBM_s115
                                 + 0.948991 w_k1_INTC_s115
                                 + 1.15935 w_k1_KO_s115 + 0.91577 w_k1_MAT_s115
                                 + 0.949293 w_k1_MSFT_s115
                                 + 1.07706 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.916986 w_k1_ADM_s115
                                 + 1.14469 w_k1_AXP_s115
                                 + 1.13561 w_k1_DIS_s115
                                 + 1.04985 w_k1_FDX_s115
                                 + 1.20873 w_k1_IBM_s115
                                 + 1.0632 w_k1_INTC_s115 + 1.16939 w_k1_KO_s115
                                 + 1.35517 w_k1_MAT_s115
                                 + 1.13418 w_k1_MSFT_s115
                                 + 1.12938 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.701052 w_k1_ADM_s115
                                 + 0.849493 w_k1_AXP_s115
                                 + 0.51148 w_k1_DIS_s115
                                 + 0.731278 w_k1_FDX_s115
                                 + 0.852755 w_k1_IBM_s115
                                 + 0.857072 w_k1_INTC_s115
                                 + 0.676361 w_k1_KO_s115
                                 + 0.693214 w_k1_MAT_s115
                                 + 0.699882 w_k1_MSFT_s115
                                 + 0.72354 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.09398 w_k1_ADM_s115
                                 + 1.61972 w_k1_AXP_s115
                                 + 1.35722 w_k1_DIS_s115
                                 + 1.33276 w_k1_FDX_s115
                                 + 1.17017 w_k1_IBM_s115
                                 + 1.13989 w_k1_INTC_s115
                                 + 1.23245 w_k1_KO_s115 + 1.17977 w_k1_MAT_s115
                                 + 1.11615 w_k1_MSFT_s115
                                 + 1.24206 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.30685 w_k1_ADM_s116
                                 + 1.21481 w_k1_AXP_s116
                                 + 1.12043 w_k1_DIS_s116
                                 + 1.26762 w_k1_FDX_s116
                                 + 0.898056 w_k1_IBM_s116
                                 + 1.27543 w_k1_INTC_s116
                                 + 0.96921 w_k1_KO_s116
                                 + 0.695531 w_k1_MAT_s116
                                 + 1.18335 w_k1_MSFT_s116
                                 + 1.06967 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2476 w_k1_ADM_s116
                                 + 1.66803 w_k1_AXP_s116
                                 + 0.997779 w_k1_DIS_s116
                                 + 1.13936 w_k1_FDX_s116
                                 + 1.03423 w_k1_IBM_s116
                                 + 1.19344 w_k1_INTC_s116
                                 + 1.08286 w_k1_KO_s116 + 1.24359 w_k1_MAT_s116
                                 + 0.959778 w_k1_MSFT_s116
                                 + 1.04965 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.927553 w_k1_ADM_s116
                                 + 0.840765 w_k1_AXP_s116
                                 + 0.637098 w_k1_DIS_s116
                                 + 0.816679 w_k1_FDX_s116
                                 + 0.700848 w_k1_IBM_s116
                                 + 0.961553 w_k1_INTC_s116
                                 + 0.726637 w_k1_KO_s116
                                 + 0.595702 w_k1_MAT_s116
                                 + 0.795567 w_k1_MSFT_s116
                                 + 0.78221 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.934759 w_k1_ADM_s116
                                 + 0.861826 w_k1_AXP_s116
                                 + 0.870373 w_k1_DIS_s116
                                 + 0.920367 w_k1_FDX_s116
                                 + 1.17862 w_k1_IBM_s116
                                 + 1.14075 w_k1_INTC_s116
                                 + 1.15432 w_k1_KO_s116 + 1.08685 w_k1_MAT_s116
                                 + 1.05644 w_k1_MSFT_s116
                                 + 1.07903 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03871 w_k1_ADM_s116
                                 + 0.752162 w_k1_AXP_s116
                                 + 0.845872 w_k1_DIS_s116
                                 + 0.886223 w_k1_FDX_s116
                                 + 0.916727 w_k1_IBM_s116
                                 + 0.827851 w_k1_INTC_s116
                                 + 0.890189 w_k1_KO_s116
                                 + 0.640772 w_k1_MAT_s116
                                 + 0.852143 w_k1_MSFT_s116
                                 + 0.943102 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.887318 w_k1_ADM_s116
                                 + 0.937929 w_k1_AXP_s116
                                 + 0.717609 w_k1_DIS_s116
                                 + 0.828234 w_k1_FDX_s116
                                 + 0.882157 w_k1_IBM_s116
                                 + 0.937898 w_k1_INTC_s116
                                 + 0.803934 w_k1_KO_s116
                                 + 1.08672 w_k1_MAT_s116
                                 + 0.885952 w_k1_MSFT_s116
                                 + 0.904575 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.31736 w_k1_ADM_s116
                                 + 1.09189 w_k1_AXP_s116 + 1.1222 w_k1_DIS_s116
                                 + 1.18017 w_k1_FDX_s116
                                 + 0.779533 w_k1_IBM_s116
                                 + 1.29512 w_k1_INTC_s116
                                 + 0.945581 w_k1_KO_s116
                                 + 0.671156 w_k1_MAT_s116
                                 + 1.1439 w_k1_MSFT_s116
                                 + 0.976273 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.913945 w_k1_ADM_s116
                                 + 1.30159 w_k1_AXP_s116
                                 + 1.52668 w_k1_DIS_s116
                                 + 1.35175 w_k1_FDX_s116
                                 + 1.41335 w_k1_IBM_s116
                                 + 1.5399 w_k1_INTC_s116 + 1.33421 w_k1_KO_s116
                                 + 1.22773 w_k1_MAT_s116
                                 + 1.39038 w_k1_MSFT_s116 + 1.0263 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.30711 w_k1_ADM_s117
                                 + 1.25211 w_k1_AXP_s117
                                 + 1.51233 w_k1_DIS_s117
                                 + 1.38861 w_k1_FDX_s117
                                 + 0.906281 w_k1_IBM_s117
                                 + 1.06435 w_k1_INTC_s117
                                 + 1.14697 w_k1_KO_s117
                                 + 0.996797 w_k1_MAT_s117
                                 + 1.30524 w_k1_MSFT_s117
                                 + 1.15285 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.768148 w_k1_ADM_s117
                                 + 0.815187 w_k1_AXP_s117
                                 + 0.656192 w_k1_DIS_s117
                                 + 0.903156 w_k1_FDX_s117
                                 + 0.681662 w_k1_IBM_s117
                                 + 0.937847 w_k1_INTC_s117
                                 + 0.735525 w_k1_KO_s117
                                 + 0.791856 w_k1_MAT_s117
                                 + 0.931773 w_k1_MSFT_s117
                                 + 0.827025 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.934348 w_k1_ADM_s117
                                 + 0.934575 w_k1_AXP_s117
                                 + 0.908939 w_k1_DIS_s117
                                 + 0.944138 w_k1_FDX_s117
                                 + 1.07271 w_k1_IBM_s117
                                 + 0.95149 w_k1_INTC_s117
                                 + 0.999876 w_k1_KO_s117
                                 + 0.91427 w_k1_MAT_s117
                                 + 0.869537 w_k1_MSFT_s117
                                 + 0.856621 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.25474 w_k1_ADM_s117
                                 + 1.55264 w_k1_AXP_s117
                                 + 1.59385 w_k1_DIS_s117
                                 + 1.45998 w_k1_FDX_s117
                                 + 1.44025 w_k1_IBM_s117
                                 + 1.20231 w_k1_INTC_s117
                                 + 1.75097 w_k1_KO_s117 + 2.09117 w_k1_MAT_s117
                                 + 1.49209 w_k1_MSFT_s117
                                 + 1.44828 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.0545 w_k1_ADM_s117
                                 + 1.22981 w_k1_AXP_s117
                                 + 1.09886 w_k1_DIS_s117
                                 + 1.17261 w_k1_FDX_s117
                                 + 0.979379 w_k1_IBM_s117
                                 + 1.08437 w_k1_INTC_s117
                                 + 1.02319 w_k1_KO_s117 + 1.03087 w_k1_MAT_s117
                                 + 1.10939 w_k1_MSFT_s117
                                 + 1.09892 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.768896 w_k1_ADM_s117
                                 + 0.630236 w_k1_AXP_s117
                                 + 0.728409 w_k1_DIS_s117
                                 + 0.753381 w_k1_FDX_s117
                                 + 0.89611 w_k1_IBM_s117
                                 + 0.804072 w_k1_INTC_s117
                                 + 0.803351 w_k1_KO_s117
                                 + 0.562223 w_k1_MAT_s117
                                 + 0.788788 w_k1_MSFT_s117
                                 + 0.920807 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.976958 w_k1_ADM_s117
                                 + 0.966122 w_k1_AXP_s117
                                 + 0.838026 w_k1_DIS_s117
                                 + 0.928729 w_k1_FDX_s117
                                 + 0.878912 w_k1_IBM_s117
                                 + 1.04912 w_k1_INTC_s117
                                 + 0.883386 w_k1_KO_s117
                                 + 0.858888 w_k1_MAT_s117
                                 + 0.968222 w_k1_MSFT_s117
                                 + 0.908005 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.884401 w_k1_ADM_s117
                                 + 1.01223 w_k1_AXP_s117
                                 + 0.975303 w_k1_DIS_s117
                                 + 0.871996 w_k1_FDX_s117
                                 + 1.40509 w_k1_IBM_s117
                                 + 0.880057 w_k1_INTC_s117
                                 + 1.25136 w_k1_KO_s117 + 1.3999 w_k1_MAT_s117
                                 + 1.07324 w_k1_MSFT_s117
                                 + 1.07381 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0170399981006835 w_k1_ADM_s000
                                 + 0.0146100047573556 w_k1_ADM_s001
                                 + 0.0137515891472185 w_k1_ADM_s002
                                 + 0.0167173343820149 w_k1_ADM_s003
                                 + 0.0258998388046469 w_k1_ADM_s004
                                 + 0.0297252361454362 w_k1_ADM_s005
                                 + 0.027769136271705 w_k1_ADM_s006
                                 + 0.0142314896979912 w_k1_ADM_s007
                                 + 0.00975179396555653 w_k1_ADM_s010
                                 + 0.008628978398507 w_k1_ADM_s011
                                 + 0.0187734014140611 w_k1_ADM_s012
                                 + 0.0178933954303491 w_k1_ADM_s013
                                 + 0.0319002451823423 w_k1_ADM_s014
                                 + 0.0150868989109033 w_k1_ADM_s015
                                 + 0.0354939536500387 w_k1_ADM_s016
                                 + 0.017073612611295 w_k1_ADM_s017
                                 + 0.0357976424434647 w_k1_ADM_s100
                                 + 0.0412783948344923 w_k1_ADM_s101
                                 + 0.0511187706970176 w_k1_ADM_s102
                                 + 0.0419848794828475 w_k1_ADM_s103
                                 + 0.0433189295680111 w_k1_ADM_s104
                                 + 0.0508857037799438 w_k1_ADM_s105
                                 + 0.139680930683626 w_k1_ADM_s106
                                 + 0.0473429479005594 w_k1_ADM_s107
                                 + 0.0312693301743058 w_k1_ADM_s110
                                 + 0.0172708311450132 w_k1_ADM_s111
                                 + 0.0264844284614333 w_k1_ADM_s112
                                 + 0.0279764234005938 w_k1_ADM_s113
                                 + 0.0653899218835291 w_k1_ADM_s114
                                 + 0.0197525988238933 w_k1_ADM_s115
                                 + 0.0271193582107079 w_k1_ADM_s116
                                 + 0.0371562400988458 w_k1_ADM_s117
                                 + 0.0198266001287532 w_k1_AXP_s000
                                 + 0.0122233212729641 w_k1_AXP_s001
                                 + 0.0166213223025367 w_k1_AXP_s002
                                 + 0.0199580717331809 w_k1_AXP_s003
                                 + 0.0328545916948359 w_k1_AXP_s004
                                 + 0.0311684153996821 w_k1_AXP_s005
                                 + 0.026668010374864 w_k1_AXP_s006
                                 + 0.0159893110819694 w_k1_AXP_s007
                                 + 0.011328601396736 w_k1_AXP_s010
                                 + 0.00786297009801814 w_k1_AXP_s011
                                 + 0.0181029683155729 w_k1_AXP_s012
                                 + 0.0167970774179342 w_k1_AXP_s013
                                 + 0.0269821832160544 w_k1_AXP_s014
                                 + 0.0141940181134535 w_k1_AXP_s015
                                 + 0.0354020998871623 w_k1_AXP_s016
                                 + 0.0158242616047076 w_k1_AXP_s017
                                 + 0.0305572728393861 w_k1_AXP_s100
                                 + 0.040221510715513 w_k1_AXP_s101
                                 + 0.0469580154200002 w_k1_AXP_s102
                                 + 0.0442409246960051 w_k1_AXP_s103
                                 + 0.0390503816577717 w_k1_AXP_s104
                                 + 0.0468776986492708 w_k1_AXP_s105
                                 + 0.146544260103581 w_k1_AXP_s106
                                 + 0.0482395168634044 w_k1_AXP_s107
                                 + 0.0266011206731783 w_k1_AXP_s110
                                 + 0.0218972504090997 w_k1_AXP_s111
                                 + 0.0251061735067005 w_k1_AXP_s112
                                 + 0.0287372074625354 w_k1_AXP_s113
                                 + 0.07717850248038 w_k1_AXP_s114
                                 + 0.0227754498584357 w_k1_AXP_s115
                                 + 0.0274762476599036 w_k1_AXP_s116
                                 + 0.0389300545782 w_k1_AXP_s117
                                 + 0.0213122860347309 w_k1_DIS_s000
                                 + 0.0127183884449518 w_k1_DIS_s001
                                 + 0.017468676785705 w_k1_DIS_s002
                                 + 0.0176696865584738 w_k1_DIS_s003
                                 + 0.0331363008314531 w_k1_DIS_s004
                                 + 0.0327783934714309 w_k1_DIS_s005
                                 + 0.0285252200589662 w_k1_DIS_s006
                                 + 0.0146007490060766 w_k1_DIS_s007
                                 + 0.0118252750821116 w_k1_DIS_s010
                                 + 0.00910512337275555 w_k1_DIS_s011
                                 + 0.0184468677592948 w_k1_DIS_s012
                                 + 0.0161131475289631 w_k1_DIS_s013
                                 + 0.0254052587695574 w_k1_DIS_s014
                                 + 0.0135445401355075 w_k1_DIS_s015
                                 + 0.0360513636512125 w_k1_DIS_s016
                                 + 0.0150966048902256 w_k1_DIS_s017
                                 + 0.031689162862316 w_k1_DIS_s100
                                 + 0.0409449776518383 w_k1_DIS_s101
                                 + 0.0560058810534932 w_k1_DIS_s102
                                 + 0.0422515245342227 w_k1_DIS_s103
                                 + 0.0430129132270167 w_k1_DIS_s104
                                 + 0.0488603792792727 w_k1_DIS_s105
                                 + 0.142660085395167 w_k1_DIS_s106
                                 + 0.0475629841650675 w_k1_DIS_s107
                                 + 0.0276049297671083 w_k1_DIS_s110
                                 + 0.0220271106256383 w_k1_DIS_s111
                                 + 0.0265800692358793 w_k1_DIS_s112
                                 + 0.0296312472764278 w_k1_DIS_s113
                                 + 0.0736440893091599 w_k1_DIS_s114
                                 + 0.0203150073855593 w_k1_DIS_s115
                                 + 0.0262484693577291 w_k1_DIS_s116
                                 + 0.0382290801860203 w_k1_DIS_s117
                                 + 0.020130343294604 w_k1_FDX_s000
                                 + 0.0121604951915353 w_k1_FDX_s001
                                 + 0.0158828543164548 w_k1_FDX_s002
                                 + 0.0175249142704245 w_k1_FDX_s003
                                 + 0.0307862355024931 w_k1_FDX_s004
                                 + 0.0298702104430099 w_k1_FDX_s005
                                 + 0.0278358857054653 w_k1_FDX_s006
                                 + 0.0142290734419918 w_k1_FDX_s007
                                 + 0.0111759211323524 w_k1_FDX_s010
                                 + 0.0082800083886133 w_k1_FDX_s011
                                 + 0.0177904090087275 w_k1_FDX_s012
                                 + 0.0163797999022318 w_k1_FDX_s013
                                 + 0.0268160782203061 w_k1_FDX_s014
                                 + 0.0141939338513233 w_k1_FDX_s015
                                 + 0.0363950878605991 w_k1_FDX_s016
                                 + 0.0146666020465905 w_k1_FDX_s017
                                 + 0.0312024159225328 w_k1_FDX_s100
                                 + 0.0419309466278502 w_k1_FDX_s101
                                 + 0.0522024832766839 w_k1_FDX_s102
                                 + 0.0414140438301073 w_k1_FDX_s103
                                 + 0.0408636350766588 w_k1_FDX_s104
                                 + 0.0466489668660944 w_k1_FDX_s105
                                 + 0.138125732669168 w_k1_FDX_s106
                                 + 0.0455126257380279 w_k1_FDX_s107
                                 + 0.0255568032840485 w_k1_FDX_s110
                                 + 0.0203380694683061 w_k1_FDX_s111
                                 + 0.0246505010349222 w_k1_FDX_s112
                                 + 0.0284851338143566 w_k1_FDX_s113
                                 + 0.072301459601336 w_k1_FDX_s114
                                 + 0.0206256303965887 w_k1_FDX_s115
                                 + 0.0274912793432735 w_k1_FDX_s116
                                 + 0.0391041737573836 w_k1_FDX_s117
                                 + 0.0194364855781082 w_k1_IBM_s000
                                 + 0.013206721029136 w_k1_IBM_s001
                                 + 0.0153571443819592 w_k1_IBM_s002
                                 + 0.0176697586330478 w_k1_IBM_s003
                                 + 0.0326432425759369 w_k1_IBM_s004
                                 + 0.0314806968628768 w_k1_IBM_s005
                                 + 0.0282962750245638 w_k1_IBM_s006
                                 + 0.0138919499094144 w_k1_IBM_s007
                                 + 0.01150742332634 w_k1_IBM_s010
                                 + 0.00902953636155304 w_k1_IBM_s011
                                 + 0.0177918423597986 w_k1_IBM_s012
                                 + 0.0168466983949255 w_k1_IBM_s013
                                 + 0.0264039660315939 w_k1_IBM_s014
                                 + 0.0143420764920313 w_k1_IBM_s015
                                 + 0.0367918935089633 w_k1_IBM_s016
                                 + 0.0146880533227923 w_k1_IBM_s017
                                 + 0.0314134589722252 w_k1_IBM_s100
                                 + 0.0416462234913213 w_k1_IBM_s101
                                 + 0.052393305706635 w_k1_IBM_s102
                                 + 0.0445696694867581 w_k1_IBM_s103
                                 + 0.0427401420909396 w_k1_IBM_s104
                                 + 0.045324047290914 w_k1_IBM_s105
                                 + 0.144799823777924 w_k1_IBM_s106
                                 + 0.0462630168051092 w_k1_IBM_s107
                                 + 0.0268962489449343 w_k1_IBM_s110
                                 + 0.0202074814952461 w_k1_IBM_s111
                                 + 0.0245966783353711 w_k1_IBM_s112
                                 + 0.0298660985235834 w_k1_IBM_s113
                                 + 0.0761429600804198 w_k1_IBM_s114
                                 + 0.0214072369468992 w_k1_IBM_s115
                                 + 0.0277543641327087 w_k1_IBM_s116
                                 + 0.0379242689308316 w_k1_IBM_s117
                                 + 0.0219253108488299 w_k1_INTC_s000
                                 + 0.0128911089736884 w_k1_INTC_s001
                                 + 0.0142277327922953 w_k1_INTC_s002
                                 + 0.0189180579493537 w_k1_INTC_s003
                                 + 0.0332988154911565 w_k1_INTC_s004
                                 + 0.030609013360759 w_k1_INTC_s005
                                 + 0.0303645175840237 w_k1_INTC_s006
                                 + 0.0160974602949591 w_k1_INTC_s007
                                 + 0.0098862381028937 w_k1_INTC_s010
                                 + 0.00864297862177751 w_k1_INTC_s011
                                 + 0.0207620981789585 w_k1_INTC_s012
                                 + 0.0137522339749204 w_k1_INTC_s013
                                 + 0.0244682449178132 w_k1_INTC_s014
                                 + 0.0146113227635642 w_k1_INTC_s015
                                 + 0.0339255242227284 w_k1_INTC_s016
                                 + 0.014085638472247 w_k1_INTC_s017
                                 + 0.0275604669695502 w_k1_INTC_s100
                                 + 0.0422878611100004 w_k1_INTC_s101
                                 + 0.0507007786387712 w_k1_INTC_s102
                                 + 0.0415664721147297 w_k1_INTC_s103
                                 + 0.040014154210991 w_k1_INTC_s104
                                 + 0.0424571554807054 w_k1_INTC_s105
                                 + 0.143234221326976 w_k1_INTC_s106
                                 + 0.0536919560991738 w_k1_INTC_s107
                                 + 0.0255047507208963 w_k1_INTC_s110
                                 + 0.0215511503618491 w_k1_INTC_s111
                                 + 0.0250786603107531 w_k1_INTC_s112
                                 + 0.0275495155877173 w_k1_INTC_s113
                                 + 0.0721606774899582 w_k1_INTC_s114
                                 + 0.0197156811730005 w_k1_INTC_s115
                                 + 0.0310988775420246 w_k1_INTC_s116
                                 + 0.0374440479756455 w_k1_INTC_s117
                                 + 0.0208300726656546 w_k1_KO_s000
                                 + 0.0118494273748905 w_k1_KO_s001
                                 + 0.0146084850368181 w_k1_KO_s002
                                 + 0.0166700203694805 w_k1_KO_s003
                                 + 0.031820758348821 w_k1_KO_s004
                                 + 0.0281956979404034 w_k1_KO_s005
                                 + 0.0295642456904417 w_k1_KO_s006
                                 + 0.0132482307727292 w_k1_KO_s007
                                 + 0.0105955501166236 w_k1_KO_s010
                                 + 0.00914189823540997 w_k1_KO_s011
                                 + 0.0174711742337095 w_k1_KO_s012
                                 + 0.0176829251329634 w_k1_KO_s013
                                 + 0.0245504309976304 w_k1_KO_s014
                                 + 0.0138407354132355 w_k1_KO_s015
                                 + 0.0325833733800001 w_k1_KO_s016
                                 + 0.0160151102292765 w_k1_KO_s017
                                 + 0.0301240870845394 w_k1_KO_s100
                                 + 0.0391278012981678 w_k1_KO_s101
                                 + 0.0540789645383246 w_k1_KO_s102
                                 + 0.0408034592003174 w_k1_KO_s103
                                 + 0.0385117868842156 w_k1_KO_s104
                                 + 0.0435129901116684 w_k1_KO_s105
                                 + 0.129743884782912 w_k1_KO_s106
                                 + 0.0449224634777219 w_k1_KO_s107
                                 + 0.0251953820940539 w_k1_KO_s110
                                 + 0.0205384879143102 w_k1_KO_s111
                                 + 0.0239976470543996 w_k1_KO_s112
                                 + 0.0296828196005345 w_k1_KO_s113
                                 + 0.0769594056193833 w_k1_KO_s114
                                 + 0.0212637398118085 w_k1_KO_s115
                                 + 0.0277120862001764 w_k1_KO_s116
                                 + 0.0390054689433247 w_k1_KO_s117
                                 + 0.0186624139438867 w_k1_MAT_s000
                                 + 0.00995394103683719 w_k1_MAT_s001
                                 + 0.014476850602139 w_k1_MAT_s002
                                 + 0.0160687374777994 w_k1_MAT_s003
                                 + 0.0333372104950701 w_k1_MAT_s004
                                 + 0.0306670647564757 w_k1_MAT_s005
                                 + 0.0360479156573744 w_k1_MAT_s006
                                 + 0.0134724551300374 w_k1_MAT_s007
                                 + 0.0107348958783125 w_k1_MAT_s010
                                 + 0.00994512493552312 w_k1_MAT_s011
                                 + 0.0178586694579471 w_k1_MAT_s012
                                 + 0.0163645344268005 w_k1_MAT_s013
                                 + 0.0290884201606886 w_k1_MAT_s014
                                 + 0.0142242034234472 w_k1_MAT_s015
                                 + 0.0289121460386435 w_k1_MAT_s016
                                 + 0.0174441681424211 w_k1_MAT_s017
                                 + 0.0284788567051681 w_k1_MAT_s100
                                 + 0.0408904522402019 w_k1_MAT_s101
                                 + 0.0471397690116244 w_k1_MAT_s102
                                 + 0.0374695452817063 w_k1_MAT_s103
                                 + 0.0417833991273567 w_k1_MAT_s104
                                 + 0.0426297681695192 w_k1_MAT_s105
                                 + 0.146084906602791 w_k1_MAT_s106
                                 + 0.0395160286266468 w_k1_MAT_s107
                                 + 0.0273791272402722 w_k1_MAT_s110
                                 + 0.0233040675229113 w_k1_MAT_s111
                                 + 0.0231836835327649 w_k1_MAT_s112
                                 + 0.0317760564428433 w_k1_MAT_s113
                                 + 0.0689699160848877 w_k1_MAT_s114
                                 + 0.0192399185905298 w_k1_MAT_s115
                                 + 0.0252582274463792 w_k1_MAT_s116
                                 + 0.0383984664471536 w_k1_MAT_s117
                                 + 0.0188832771280351 w_k1_MSFT_s000
                                 + 0.0119054674192634 w_k1_MSFT_s001
                                 + 0.0167275764155221 w_k1_MSFT_s002
                                 + 0.0169094651366098 w_k1_MSFT_s003
                                 + 0.0322208904806962 w_k1_MSFT_s004
                                 + 0.0286931386920325 w_k1_MSFT_s005
                                 + 0.0296417761927823 w_k1_MSFT_s006
                                 + 0.0134502247586667 w_k1_MSFT_s007
                                 + 0.0108261358999281 w_k1_MSFT_s010
                                 + 0.00896876919351599 w_k1_MSFT_s011
                                 + 0.0189934435079825 w_k1_MSFT_s012
                                 + 0.0160213441037457 w_k1_MSFT_s013
                                 + 0.0247064438252613 w_k1_MSFT_s014
                                 + 0.0127322278216557 w_k1_MSFT_s015
                                 + 0.0354299424807718 w_k1_MSFT_s016
                                 + 0.0137422000757609 w_k1_MSFT_s017
                                 + 0.0307509848318568 w_k1_MSFT_s100
                                 + 0.0371579962792761 w_k1_MSFT_s101
                                 + 0.0491051193071142 w_k1_MSFT_s102
                                 + 0.0428893238420306 w_k1_MSFT_s103
                                 + 0.0413181135594753 w_k1_MSFT_s104
                                 + 0.0525097879054073 w_k1_MSFT_s105
                                 + 0.144495095991909 w_k1_MSFT_s106
                                 + 0.0486107578093207 w_k1_MSFT_s107
                                 + 0.0280752463528947 w_k1_MSFT_s110
                                 + 0.0211574412734562 w_k1_MSFT_s111
                                 + 0.0244157790274313 w_k1_MSFT_s112
                                 + 0.0282159705264353 w_k1_MSFT_s113
                                 + 0.0758584802374292 w_k1_MSFT_s114
                                 + 0.019547562600134 w_k1_MSFT_s115
                                 + 0.0281074250100933 w_k1_MSFT_s116
                                 + 0.039482208468515 w_k1_MSFT_s117
                                 + 0.0194652907038377 w_k1_PG_s000
                                 + 0.0117553294796037 w_k1_PG_s001
                                 + 0.0147579641585815 w_k1_PG_s002
                                 + 0.0175661504153674 w_k1_PG_s003
                                 + 0.0304149478305791 w_k1_PG_s004
                                 + 0.0306909541417056 w_k1_PG_s005
                                 + 0.0296448637235441 w_k1_PG_s006
                                 + 0.0140810223052477 w_k1_PG_s007
                                 + 0.0107637400614415 w_k1_PG_s010
                                 + 0.00865963071816053 w_k1_PG_s011
                                 + 0.0193596605617335 w_k1_PG_s012
                                 + 0.0168070127429808 w_k1_PG_s013
                                 + 0.025548439046737 w_k1_PG_s014
                                 + 0.0132289188600257 w_k1_PG_s015
                                 + 0.0349970368105794 w_k1_PG_s016
                                 + 0.01482744172202 w_k1_PG_s017
                                 + 0.0290189444015112 w_k1_PG_s100
                                 + 0.0387386025307953 w_k1_PG_s101
                                 + 0.0522294449442554 w_k1_PG_s102
                                 + 0.0389333604956403 w_k1_PG_s103
                                 + 0.0389551561092033 w_k1_PG_s104
                                 + 0.0462974706031533 w_k1_PG_s105
                                 + 0.137114424581113 w_k1_PG_s106
                                 + 0.0491285746387968 w_k1_PG_s107
                                 + 0.0272092461432811 w_k1_PG_s110
                                 + 0.0198566875397458 w_k1_PG_s111
                                 + 0.026817767331646 w_k1_PG_s112
                                 + 0.0290538038814721 w_k1_PG_s113
                                 + 0.07415059990307 w_k1_PG_s114
                                 + 0.0207804800785022 w_k1_PG_s115
                                 + 0.0264086011359092 w_k1_PG_s116
                                 + 0.0385919604964802 w_k1_PG_s117 >= 1.05
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
