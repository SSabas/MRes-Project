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
 w_k1_ADM_s0_return_constraint:  1.08762 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.983814 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.816552 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.18525 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.80617 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.09687 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.712756 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.4429 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.909465 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.0685 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.79852 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.19106 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.23558 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.19915 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.743415 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.740432 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.14388 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.21191 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.838669 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.04851 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.21383 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.830183 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.09658 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.19672 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.979428 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.92099 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.872134 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.19444 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.19449 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.900413 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.08249 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.753992 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.816291 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.806016 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.08304 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.12071 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.873759 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.753794 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.10704 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.966061 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.790853 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.21319 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.702678 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12538 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.843033 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.42017 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.778125 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.912104 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.772023 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.34241 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.39184 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.03756 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.65102 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.706437 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.52666 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.5704 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.857583 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.11426 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.24377 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.890653 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.00767 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.14186 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.951802 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.730227 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.827478 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.41219 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.33475 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.18776 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.14242 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.702838 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.63017 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.802475 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.913418 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.58647 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.34599 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.974002 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.15101 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.902379 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.757814 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.32267 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.691547 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.25136 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.653109 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.3223 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.694253 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.934431 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.873228 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.31447 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.55025 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.30251 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.559682 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.631748 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.35531 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.70593 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.851741 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.09036 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.42827 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.792008 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.1321 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.33006 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.86837 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.96959 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.726484 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.50989 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.67656 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.03354 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.08445 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.623618 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.605756 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.668272 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.02119 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.81474 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.45687 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.803691 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1104 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.886711 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.874511 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.17465 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.803063 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18597 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.702444 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.24283 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.869799 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.929521 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.778757 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12845 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.19154 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.22215 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.683263 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.779968 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.36223 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.28764 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.97197 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.05416 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.44363 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.815762 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.01193 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.34332 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.976294 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.885952 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.808027 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.23046 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.3507 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.902504 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06328 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.689209 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.822411 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.810204 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.07332 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.3628 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.09915 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.854547 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.951119 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.10573 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.799659 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.2669 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.766602 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.14271 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.05684 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.969953 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.652573 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.984163 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.26502 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.37157 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.75067 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.785771 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.954039 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.6297 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.974254 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.82197 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.692777 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.16577 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.922977 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.09259 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.44846 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.841481 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.698795 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.912745 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.20964 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.67124 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.20841 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 2.04028 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.08572 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.985856 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.481104 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.684605 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.648922 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.29395 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.93964 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.0821 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.04183 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.997432 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.868152 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.05847 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.737882 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.02768 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.890725 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08063 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.903529 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.991909 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04844 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.18259 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.26002 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.16846 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.798482 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.899521 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.30616 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.19587 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.996677 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.33291 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.37645 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.895666 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.0929 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.23185 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.13895 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.933984 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.938181 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.52637 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.40152 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.08866 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.0291 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.792117 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.765292 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.936931 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.998986 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.25874 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.08306 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.849203 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.02243 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.02973 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.811752 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.3366 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.717844 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.08148 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.830113 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0623 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.731784 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.935825 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.943597 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.26936 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.47534 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.947817 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.666187 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.586594 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.07681 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.56353 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.725299 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.01874 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.06284 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.907087 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.19937 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.0579 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.901087 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.944222 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.03382 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.68653 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.41755 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.48847 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.1038 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.831029 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.589744 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.758973 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.885524 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.40992 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.69309 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.885676 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.0672 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.10715 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.739498 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.26073 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.75793 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.10115 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.07099 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.07921 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.505199 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.663254 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.999278 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.45032 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.53734 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.622311 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.721119 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.614065 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.09247 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 2.41073 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.514125 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.998758 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.914239 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.19579 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.32063 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.708289 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.638182 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.751342 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.950692 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.49626 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.2233 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 2.01186 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.28146 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.875644 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.411565 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.729977 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.567495 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.43409 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 2.25832 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.1639 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.08128 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.916925 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.891034 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.17902 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.823582 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.13107 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.727177 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13283 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.818007 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.912148 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.878811 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.09079 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.187 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.26901 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.698213 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.720594 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.19938 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1819 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.901817 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.948237 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.28332 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.822025 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.06375 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.20505 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.989657 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.934037 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.915916 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.24396 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.29512 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.02579 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07447 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.715778 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.86337 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.843858 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.06848 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.42004 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.14589 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.826195 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.05706 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.959023 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.846739 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.15519 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.853783 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11353 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.890155 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11456 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.828239 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.965446 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.965494 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.16899 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.33512 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.08102 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.80465 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.774857 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.17207 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.19653 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.901418 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03163 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.11596 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.925051 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0588 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.04057 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.983538 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.921434 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.974951 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.17408 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.24172 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.20863 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.05944 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.787693 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.733743 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.796809 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.923428 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.20448 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.19708 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.884437 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.372002233988727 z_k1_s0000
                                 - 0.209000693914727 z_k1_s0001
                                 - 0.27000067610216 z_k1_s0002
                                 - 0.354000235209956 z_k1_s0003
                                 - 0.697001109615705 z_k1_s0004
                                 - 0.377000928533344 z_k1_s0005
                                 - 0.214000504862369 z_k1_s0006
                                 - 0.298001459473946 z_k1_s0007
                                 - 0.25799974152251 z_k1_s0010
                                 - 0.723998929869626 z_k1_s0011
                                 - 0.24199967545916 z_k1_s0012
                                 - 0.356000234410464 z_k1_s0013
                                 - 0.850997645123818 z_k1_s0014
                                 - 0.326999894045806 z_k1_s0015
                                 - 0.397001153822239 z_k1_s0016
                                 - 0.37499868023691 z_k1_s0017
                                 - 0.470001829957734 z_k1_s0020
                                 - 0.515001029335291 z_k1_s0021
                                 - 0.609000521725608 z_k1_s0022
                                 - 0.79499895968864 z_k1_s0023
                                 - 0.483001016265986 z_k1_s0024
                                 - 0.692001374625273 z_k1_s0025
                                 - 0.405000656809092 z_k1_s0026
                                 - 0.399000763558751 z_k1_s0027
                                 - 0.480997810546114 z_k1_s0030
                                 - 1.14399817202154 z_k1_s0031
                                 - 0.824997034182435 z_k1_s0032
                                 - 0.277999275662113 z_k1_s0033
                                 - 0.65599830409762 z_k1_s0034
                                 - 0.294999178061706 z_k1_s0035
                                 - 0.542997509073928 z_k1_s0036
                                 - 0.408999340798447 z_k1_s0037
                                 - 0.228001080385239 z_k1_s0040
                                 - 0.299999858400805 z_k1_s0041
                                 - 0.420999702289078 z_k1_s0042
                                 - 0.159000023952479 z_k1_s0043
                                 - 0.2880010520654 z_k1_s0044
                                 - 0.186000110208604 z_k1_s0045
                                 - 0.142000130976486 z_k1_s0046
                                 - 0.256000275168896 z_k1_s0047
                                 - 0.613999379542234 z_k1_s0050
                                 - 0.663001063859666 z_k1_s0051
                                 - 0.309000014118607 z_k1_s0052
                                 - 0.622999049907061 z_k1_s0053
                                 - 0.693998934450245 z_k1_s0054
                                 - 0.863999666003719 z_k1_s0055
                                 - 0.247999962113994 z_k1_s0056
                                 - 0.457999129222313 z_k1_s0057
                                 - 0.109000136640101 z_k1_s0060
                                 - 0.168000148122311 z_k1_s0061
                                 - 0.283000075445192 z_k1_s0062
                                 - 0.222000667869121 z_k1_s0063
                                 - 0.266000601743934 z_k1_s0064
                                 - 0.383999422907455 z_k1_s0065
                                 - 0.267000099490714 z_k1_s0066
                                 - 0.303999543135069 z_k1_s0067
                                 - 0.207999240489738 z_k1_s0070
                                 - 0.764997740609002 z_k1_s0071
                                 - 0.346997655223083 z_k1_s0072
                                 - 0.350999133043402 z_k1_s0073
                                 - 0.209999113903613 z_k1_s0074
                                 - 0.525996711720304 z_k1_s0075
                                 - 0.291000045639538 z_k1_s0076
                                 - 0.186999127150243 z_k1_s0077
                                 - 0.438000238783298 z_k1_s0100
                                 - 0.522999069952416 z_k1_s0101
                                 - 0.206999958021017 z_k1_s0102
                                 - 0.104999922119409 z_k1_s0103
                                 - 0.352999352616085 z_k1_s0104
                                 - 0.138999865586675 z_k1_s0105
                                 - 0.138999865586675 z_k1_s0106
                                 - 0.150999821457498 z_k1_s0107
                                 - 0.292999380056287 z_k1_s0110
                                 - 0.151000054411375 z_k1_s0111
                                 - 0.315000520818557 z_k1_s0112
                                 - 0.177000109602752 z_k1_s0113
                                 - 0.209999706545508 z_k1_s0114
                                 - 0.311000837589215 z_k1_s0115
                                 - 0.339999982502392 z_k1_s0116
                                 - 0.124999999219999 z_k1_s0117
                                 - 0.47099740294103 z_k1_s0120
                                 - 0.753999311250283 z_k1_s0121
                                 - 0.261999520024229 z_k1_s0122
                                 - 0.234999715881037 z_k1_s0123
                                 - 0.386000205005207 z_k1_s0124
                                 - 0.296999689572125 z_k1_s0125
                                 - 0.27699969805887 z_k1_s0126
                                 - 1.00599994191609 z_k1_s0127
                                 - 0.101999827598018 z_k1_s0130
                                 - 0.391999104871013 z_k1_s0131
                                 - 0.139999892110895 z_k1_s0132
                                 - 0.147999783071606 z_k1_s0133
                                 - 0.258999408575491 z_k1_s0134
                                 - 0.320999569570137 z_k1_s0135
                                 - 0.316998670990597 z_k1_s0136
                                 - 0.438999602689214 z_k1_s0137
                                 - 0.562000909990845 z_k1_s0140
                                 - 0.698999320636019 z_k1_s0141
                                 - 0.219999815068394 z_k1_s0142
                                 - 0.460999404561216 z_k1_s0143
                                 - 0.577001337579234 z_k1_s0144
                                 - 0.506000687326386 z_k1_s0145
                                 - 0.588000277478524 z_k1_s0146
                                 - 0.507999370672641 z_k1_s0147
                                 - 0.841998637546946 z_k1_s0150
                                 - 1.15199484664233 z_k1_s0151
                                 - 0.481999017410833 z_k1_s0152
                                 - 0.482999216891422 z_k1_s0153
                                 - 0.40499940857841 z_k1_s0154
                                 - 1.11599685352534 z_k1_s0155
                                 - 1.14099527755134 z_k1_s0156
                                 - 0.941998896639685 z_k1_s0157
                                 - 0.163999863080722 z_k1_s0160
                                 - 0.259999504701396 z_k1_s0161
                                 - 0.168999853555748 z_k1_s0162
                                 - 0.192000336240395 z_k1_s0163
                                 - 0.263000551985459 z_k1_s0164
                                 - 0.26599984427111 z_k1_s0165
                                 - 0.258000561510432 z_k1_s0166
                                 - 0.183000704385029 z_k1_s0167
                                 - 0.319999577953446 z_k1_s0170
                                 - 1.04899996662591 z_k1_s0171
                                 - 0.487001239586341 z_k1_s0172
                                 - 0.252999477685006 z_k1_s0173
                                 - 0.461999542171962 z_k1_s0174
                                 - 0.342999314425097 z_k1_s0175
                                 - 0.663997971652431 z_k1_s0176
                                 - 1.17499707638557 z_k1_s0177
                                 - 0.115000103787404 z_k1_s1000
                                 - 0.141000261269066 z_k1_s1001
                                 - 0.18099982885542 z_k1_s1002
                                 - 0.0990001514527099 z_k1_s1003
                                 - 0.329000609627822 z_k1_s1004
                                 - 0.176000352782697 z_k1_s1005
                                 - 0.0890000716058932 z_k1_s1006
                                 - 0.123000142604827 z_k1_s1007
                                 - 0.139000164368222 z_k1_s1010
                                 - 0.157000319293685 z_k1_s1011
                                 - 0.0840000005850528 z_k1_s1012
                                 - 0.116999975014879 z_k1_s1013
                                 - 0.0620000176318351 z_k1_s1014
                                 - 0.140000172975192 z_k1_s1015
                                 - 0.160000345114595 z_k1_s1016
                                 - 0.173000457005207 z_k1_s1017
                                 - 0.359000786470763 z_k1_s1020
                                 - 1.17700167797798 z_k1_s1021
                                 - 0.422002878457335 z_k1_s1022
                                 - 0.34000100875151 z_k1_s1023
                                 - 0.263000729117954 z_k1_s1024
                                 - 0.516999833548859 z_k1_s1025
                                 - 0.468000956321606 z_k1_s1026
                                 - 0.321000844331307 z_k1_s1027
                                 - 0.271000157479605 z_k1_s1030
                                 - 0.483000311643383 z_k1_s1031
                                 - 0.262000255929821 z_k1_s1032
                                 - 0.583001244976114 z_k1_s1033
                                 - 0.540001390993523 z_k1_s1034
                                 - 0.384999032076984 z_k1_s1035
                                 - 0.680998875539289 z_k1_s1036
                                 - 0.443999765193457 z_k1_s1037
                                 - 0.294998424271119 z_k1_s1040
                                 - 0.261999932467255 z_k1_s1041
                                 - 0.129999611671089 z_k1_s1042
                                 - 0.125999698216368 z_k1_s1043
                                 - 0.215000155176688 z_k1_s1044
                                 - 0.146999620052514 z_k1_s1045
                                 - 0.10799966967139 z_k1_s1046
                                 - 0.388999650847176 z_k1_s1047
                                 - 0.086999783033125 z_k1_s1050
                                 - 0.0700000186060494 z_k1_s1051
                                 - 0.0700000186060494 z_k1_s1052
                                 - 0.0970004275199915 z_k1_s1053
                                 - 0.0750000080493333 z_k1_s1054
                                 - 0.160000327785384 z_k1_s1055
                                 - 0.17200026916925 z_k1_s1056
                                 - 0.100999853314365 z_k1_s1057
                                 - 0.0550000158505489 z_k1_s1060
                                 - 0.14500022851528 z_k1_s1061
                                 - 0.136999890100301 z_k1_s1062
                                 - 0.0710000764798585 z_k1_s1063
                                 - 0.142000358359933 z_k1_s1064
                                 - 0.0730000583834951 z_k1_s1065
                                 - 0.172000086914483 z_k1_s1066
                                 - 0.232000571024664 z_k1_s1067
                                 - 0.224000617033865 z_k1_s1070
                                 - 0.0900000190133777 z_k1_s1071
                                 - 0.10300047201576 z_k1_s1072
                                 - 0.0810000998121226 z_k1_s1073
                                 - 0.0580000490086579 z_k1_s1074
                                 - 0.115000254017323 z_k1_s1075
                                 - 0.0950000660141668 z_k1_s1076
                                 - 0.0610000772091314 z_k1_s1077
                                 - 1.27400090760126 z_k1_s1100
                                 - 0.502999157907623 z_k1_s1101
                                 - 1.38099540511757 z_k1_s1102
                                 - 1.00800034170715 z_k1_s1103
                                 - 1.12099749170517 z_k1_s1104
                                 - 0.586998615517553 z_k1_s1105
                                 - 0.858995839803872 z_k1_s1106
                                 - 1.82999552334631 z_k1_s1107
                                 - 0.31400022882192 z_k1_s1110
                                 - 0.836002285384888 z_k1_s1111
                                 - 0.337999168175377 z_k1_s1112
                                 - 0.265000300506506 z_k1_s1113
                                 - 0.716000976610951 z_k1_s1114
                                 - 0.31600036083493 z_k1_s1115
                                 - 0.215000306184586 z_k1_s1116
                                 - 0.306000361970546 z_k1_s1117
                                 - 0.393000757094209 z_k1_s1120
                                 - 0.603999605204435 z_k1_s1121
                                 - 0.229999846432155 z_k1_s1122
                                 - 0.291999469801064 z_k1_s1123
                                 - 0.286999913400048 z_k1_s1124
                                 - 0.308998937563815 z_k1_s1125
                                 - 0.158999801542295 z_k1_s1126
                                 - 0.165999912503191 z_k1_s1127
                                 - 0.71599661760104 z_k1_s1130
                                 - 0.451998953252533 z_k1_s1131
                                 - 0.269998837551933 z_k1_s1132
                                 - 0.585998676092764 z_k1_s1133
                                 - 0.281998935767025 z_k1_s1134
                                 - 0.187999415410892 z_k1_s1135
                                 - 0.62799864514696 z_k1_s1136
                                 - 0.624997309147994 z_k1_s1137
                                 - 0.364999798581381 z_k1_s1140
                                 - 0.276999292531693 z_k1_s1141
                                 - 0.222999902151087 z_k1_s1142
                                 - 0.415999763235219 z_k1_s1143
                                 - 0.219999950624385 z_k1_s1144
                                 - 0.493000534653656 z_k1_s1145
                                 - 0.303000099296292 z_k1_s1146
                                 - 0.332000594687626 z_k1_s1147
                                 - 0.679004331445734 z_k1_s1150
                                 - 0.505002897362674 z_k1_s1151
                                 - 0.583001225046244 z_k1_s1152
                                 - 0.1680005871806 z_k1_s1153
                                 - 0.3240013455634 z_k1_s1154
                                 - 0.526002877509893 z_k1_s1155
                                 - 0.658004351298515 z_k1_s1156
                                 - 0.287001096350548 z_k1_s1157
                                 - 0.271999830921558 z_k1_s1160
                                 - 0.566999428478576 z_k1_s1161
                                 - 0.319998472144106 z_k1_s1162
                                 - 0.337998633302008 z_k1_s1163
                                 - 0.509999859477777 z_k1_s1164
                                 - 0.239999701307381 z_k1_s1165
                                 - 0.29899922545911 z_k1_s1166
                                 - 0.277999696374348 z_k1_s1167
                                 - 0.674999885666073 z_k1_s1170
                                 - 0.407000736657986 z_k1_s1171
                                 - 0.607000316536077 z_k1_s1172
                                 - 0.311000819132272 z_k1_s1173
                                 - 0.677001187525372 z_k1_s1174
                                 - 0.718000562588342 z_k1_s1175
                                 - 0.967999416684759 z_k1_s1176
                                 - 0.603002678827054 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.47207 w_k1_ADM_s000
                                 + 1.11118 w_k1_AXP_s000
                                 + 1.24975 w_k1_DIS_s000
                                 + 1.27311 w_k1_FDX_s000
                                 + 0.94365 w_k1_IBM_s000
                                 + 1.09652 w_k1_INTC_s000
                                 + 1.08538 w_k1_KO_s000 + 1.08666 w_k1_MAT_s000
                                 + 1.21791 w_k1_MSFT_s000
                                 + 1.10041 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.13607 w_k1_ADM_s000
                                 + 1.41007 w_k1_AXP_s000
                                 + 0.870222 w_k1_DIS_s000
                                 + 1.12515 w_k1_FDX_s000 + 1.0694 w_k1_IBM_s000
                                 + 0.951948 w_k1_INTC_s000
                                 + 0.980852 w_k1_KO_s000
                                 + 1.19888 w_k1_MAT_s000
                                 + 0.991258 w_k1_MSFT_s000
                                 + 1.01213 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.59783 w_k1_ADM_s000
                                 + 1.74132 w_k1_AXP_s000
                                 + 1.75582 w_k1_DIS_s000
                                 + 1.48408 w_k1_FDX_s000
                                 + 1.41968 w_k1_IBM_s000
                                 + 1.04418 w_k1_INTC_s000
                                 + 1.37039 w_k1_KO_s000 + 1.52039 w_k1_MAT_s000
                                 + 1.38628 w_k1_MSFT_s000
                                 + 1.26639 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.07557 w_k1_ADM_s000
                                 + 0.945311 w_k1_AXP_s000
                                 + 1.18579 w_k1_DIS_s000
                                 + 1.02562 w_k1_FDX_s000
                                 + 1.15261 w_k1_IBM_s000
                                 + 1.08452 w_k1_INTC_s000
                                 + 1.36859 w_k1_KO_s000 + 1.16905 w_k1_MAT_s000
                                 + 1.16407 w_k1_MSFT_s000
                                 + 1.01692 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.13619 w_k1_ADM_s000
                                 + 0.848957 w_k1_AXP_s000
                                 + 0.781842 w_k1_DIS_s000
                                 + 0.849149 w_k1_FDX_s000
                                 + 0.858898 w_k1_IBM_s000
                                 + 0.946022 w_k1_INTC_s000
                                 + 0.933249 w_k1_KO_s000
                                 + 0.889141 w_k1_MAT_s000
                                 + 0.927297 w_k1_MSFT_s000
                                 + 0.943245 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.44468 w_k1_ADM_s000
                                 + 1.2889 w_k1_AXP_s000 + 1.09226 w_k1_DIS_s000
                                 + 0.99963 w_k1_FDX_s000
                                 + 1.62262 w_k1_IBM_s000
                                 + 0.847375 w_k1_INTC_s000
                                 + 1.32395 w_k1_KO_s000 + 1.42112 w_k1_MAT_s000
                                 + 0.972619 w_k1_MSFT_s000
                                 + 1.08823 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.906209 w_k1_ADM_s000
                                 + 0.588172 w_k1_AXP_s000
                                 + 0.52836 w_k1_DIS_s000
                                 + 0.635766 w_k1_FDX_s000
                                 + 0.666747 w_k1_IBM_s000
                                 + 0.570709 w_k1_INTC_s000
                                 + 0.625505 w_k1_KO_s000
                                 + 0.566689 w_k1_MAT_s000
                                 + 0.664685 w_k1_MSFT_s000
                                 + 0.684335 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.06277 w_k1_ADM_s000
                                 + 0.721023 w_k1_AXP_s000
                                 + 0.66791 w_k1_DIS_s000
                                 + 0.886036 w_k1_FDX_s000
                                 + 0.665551 w_k1_IBM_s000
                                 + 0.769438 w_k1_INTC_s000
                                 + 0.707949 w_k1_KO_s000
                                 + 0.639897 w_k1_MAT_s000
                                 + 0.865094 w_k1_MSFT_s000
                                 + 0.893216 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.889699 w_k1_ADM_s001
                                 + 1.0091 w_k1_AXP_s001
                                 + 0.790317 w_k1_DIS_s001
                                 + 0.882069 w_k1_FDX_s001
                                 + 1.43048 w_k1_IBM_s001
                                 + 0.939535 w_k1_INTC_s001
                                 + 1.19407 w_k1_KO_s001 + 1.2965 w_k1_MAT_s001
                                 + 0.793442 w_k1_MSFT_s001
                                 + 0.988153 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.875516 w_k1_ADM_s001
                                 + 0.873416 w_k1_AXP_s001
                                 + 0.859353 w_k1_DIS_s001
                                 + 0.906684 w_k1_FDX_s001
                                 + 0.958833 w_k1_IBM_s001
                                 + 1.01458 w_k1_INTC_s001
                                 + 0.972817 w_k1_KO_s001
                                 + 0.736835 w_k1_MAT_s001
                                 + 0.863367 w_k1_MSFT_s001
                                 + 0.990569 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.582715 w_k1_ADM_s001
                                 + 0.602792 w_k1_AXP_s001
                                 + 0.541765 w_k1_DIS_s001
                                 + 0.660177 w_k1_FDX_s001
                                 + 0.889295 w_k1_IBM_s001
                                 + 0.6734 w_k1_INTC_s001
                                 + 0.695734 w_k1_KO_s001
                                 + 0.979193 w_k1_MAT_s001
                                 + 0.698559 w_k1_MSFT_s001
                                 + 0.751896 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.887821 w_k1_ADM_s001
                                 + 0.960814 w_k1_AXP_s001
                                 + 1.04119 w_k1_DIS_s001
                                 + 1.01325 w_k1_FDX_s001 + 1.2585 w_k1_IBM_s001
                                 + 1.23086 w_k1_INTC_s001
                                 + 1.05741 w_k1_KO_s001 + 0.91921 w_k1_MAT_s001
                                 + 1.01343 w_k1_MSFT_s001
                                 + 0.992186 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.14358 w_k1_ADM_s001
                                 + 1.19914 w_k1_AXP_s001
                                 + 1.20208 w_k1_DIS_s001
                                 + 1.22523 w_k1_FDX_s001
                                 + 1.14523 w_k1_IBM_s001
                                 + 1.39294 w_k1_INTC_s001
                                 + 1.21455 w_k1_KO_s001 + 1.08838 w_k1_MAT_s001
                                 + 1.10153 w_k1_MSFT_s001
                                 + 1.09947 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.28431 w_k1_ADM_s001
                                 + 1.64558 w_k1_AXP_s001
                                 + 1.60619 w_k1_DIS_s001
                                 + 1.50587 w_k1_FDX_s001
                                 + 1.18906 w_k1_IBM_s001
                                 + 1.40497 w_k1_INTC_s001
                                 + 1.22146 w_k1_KO_s001 + 1.22682 w_k1_MAT_s001
                                 + 1.35422 w_k1_MSFT_s001
                                 + 1.41219 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.994226 w_k1_ADM_s001
                                 + 1.04979 w_k1_AXP_s001
                                 + 0.926329 w_k1_DIS_s001
                                 + 1.11715 w_k1_FDX_s001
                                 + 0.807148 w_k1_IBM_s001
                                 + 1.08099 w_k1_INTC_s001
                                 + 0.907681 w_k1_KO_s001
                                 + 0.819002 w_k1_MAT_s001
                                 + 1.10799 w_k1_MSFT_s001
                                 + 1.02588 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.823407 w_k1_ADM_s001
                                 + 0.803735 w_k1_AXP_s001
                                 + 0.60278 w_k1_DIS_s001
                                 + 0.779082 w_k1_FDX_s001
                                 + 0.883282 w_k1_IBM_s001
                                 + 1.02776 w_k1_INTC_s001
                                 + 0.789058 w_k1_KO_s001
                                 + 0.683658 w_k1_MAT_s001
                                 + 0.782931 w_k1_MSFT_s001
                                 + 0.82101 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.01412 w_k1_ADM_s002
                                 + 0.942551 w_k1_AXP_s002
                                 + 1.02772 w_k1_DIS_s002
                                 + 1.02884 w_k1_FDX_s002
                                 + 0.845631 w_k1_IBM_s002
                                 + 1.05695 w_k1_INTC_s002
                                 + 0.825568 w_k1_KO_s002
                                 + 0.958276 w_k1_MAT_s002
                                 + 1.02093 w_k1_MSFT_s002
                                 + 1.03767 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.953982 w_k1_ADM_s002
                                 + 0.816861 w_k1_AXP_s002
                                 + 0.847847 w_k1_DIS_s002
                                 + 0.898939 w_k1_FDX_s002
                                 + 0.672304 w_k1_IBM_s002
                                 + 0.980111 w_k1_INTC_s002
                                 + 0.766428 w_k1_KO_s002 + 1.1137 w_k1_MAT_s002
                                 + 0.991996 w_k1_MSFT_s002
                                 + 0.98681 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.11241 w_k1_ADM_s002
                                 + 1.11949 w_k1_AXP_s002
                                 + 1.12935 w_k1_DIS_s002 + 1.0142 w_k1_FDX_s002
                                 + 1.17041 w_k1_IBM_s002
                                 + 1.07619 w_k1_INTC_s002
                                 + 1.12365 w_k1_KO_s002 + 1.41434 w_k1_MAT_s002
                                 + 1.00753 w_k1_MSFT_s002
                                 + 1.11035 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.31877 w_k1_ADM_s002
                                 + 1.3073 w_k1_AXP_s002 + 1.53103 w_k1_DIS_s002
                                 + 1.35974 w_k1_FDX_s002
                                 + 1.08332 w_k1_IBM_s002
                                 + 1.24406 w_k1_INTC_s002
                                 + 1.28484 w_k1_KO_s002 + 1.21392 w_k1_MAT_s002
                                 + 1.31322 w_k1_MSFT_s002
                                 + 1.37787 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.01025 w_k1_ADM_s002
                                 + 0.970884 w_k1_AXP_s002
                                 + 1.16295 w_k1_DIS_s002
                                 + 0.939068 w_k1_FDX_s002
                                 + 1.42281 w_k1_IBM_s002
                                 + 1.09608 w_k1_INTC_s002
                                 + 1.25821 w_k1_KO_s002 + 1.32348 w_k1_MAT_s002
                                 + 1.01664 w_k1_MSFT_s002
                                 + 1.11583 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.638581 w_k1_ADM_s002
                                 + 0.646035 w_k1_AXP_s002
                                 + 0.642228 w_k1_DIS_s002
                                 + 0.703528 w_k1_FDX_s002
                                 + 0.709292 w_k1_IBM_s002
                                 + 0.829234 w_k1_INTC_s002
                                 + 0.739244 w_k1_KO_s002
                                 + 1.20042 w_k1_MAT_s002
                                 + 0.82321 w_k1_MSFT_s002
                                 + 0.776161 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.886373 w_k1_ADM_s002
                                 + 0.711117 w_k1_AXP_s002
                                 + 0.730704 w_k1_DIS_s002
                                 + 0.735303 w_k1_FDX_s002
                                 + 1.09318 w_k1_IBM_s002
                                 + 0.718196 w_k1_INTC_s002
                                 + 0.907393 w_k1_KO_s002
                                 + 1.18406 w_k1_MAT_s002
                                 + 0.872541 w_k1_MSFT_s002
                                 + 1.01845 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.30108 w_k1_ADM_s002
                                 + 1.39552 w_k1_AXP_s002
                                 + 1.20606 w_k1_DIS_s002
                                 + 1.19066 w_k1_FDX_s002
                                 + 0.963235 w_k1_IBM_s002
                                 + 0.954931 w_k1_INTC_s002
                                 + 1.00786 w_k1_KO_s002 + 1.69447 w_k1_MAT_s002
                                 + 1.16433 w_k1_MSFT_s002
                                 + 1.09789 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.779417 w_k1_ADM_s003
                                 + 0.709447 w_k1_AXP_s003
                                 + 0.649171 w_k1_DIS_s003
                                 + 0.712314 w_k1_FDX_s003
                                 + 0.950765 w_k1_IBM_s003
                                 + 0.73592 w_k1_INTC_s003
                                 + 0.814259 w_k1_KO_s003
                                 + 1.09609 w_k1_MAT_s003
                                 + 0.772357 w_k1_MSFT_s003
                                 + 0.768753 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.896541 w_k1_ADM_s003
                                 + 0.745056 w_k1_AXP_s003
                                 + 0.833427 w_k1_DIS_s003
                                 + 0.903277 w_k1_FDX_s003
                                 + 0.842165 w_k1_IBM_s003
                                 + 0.822649 w_k1_INTC_s003
                                 + 1.02242 w_k1_KO_s003
                                 + 0.989083 w_k1_MAT_s003
                                 + 0.963053 w_k1_MSFT_s003
                                 + 0.977477 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.03803 w_k1_ADM_s003
                                 + 1.14354 w_k1_AXP_s003
                                 + 1.39706 w_k1_DIS_s003
                                 + 1.11561 w_k1_FDX_s003
                                 + 1.44897 w_k1_IBM_s003
                                 + 1.17361 w_k1_INTC_s003
                                 + 1.39622 w_k1_KO_s003 + 1.88217 w_k1_MAT_s003
                                 + 1.25729 w_k1_MSFT_s003
                                 + 1.13592 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.08616 w_k1_ADM_s003
                                 + 0.887483 w_k1_AXP_s003 + 1.341 w_k1_DIS_s003
                                 + 1.21204 w_k1_FDX_s003
                                 + 0.735563 w_k1_IBM_s003
                                 + 1.10349 w_k1_INTC_s003
                                 + 1.11173 w_k1_KO_s003 + 1.35217 w_k1_MAT_s003
                                 + 1.41056 w_k1_MSFT_s003
                                 + 1.15775 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.963793 w_k1_ADM_s003
                                 + 1.15599 w_k1_AXP_s003
                                 + 1.29922 w_k1_DIS_s003
                                 + 1.19667 w_k1_FDX_s003
                                 + 0.946201 w_k1_IBM_s003
                                 + 0.994217 w_k1_INTC_s003
                                 + 1.03921 w_k1_KO_s003 + 1.60111 w_k1_MAT_s003
                                 + 1.11187 w_k1_MSFT_s003
                                 + 1.01204 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.25328 w_k1_ADM_s003
                                 + 1.10769 w_k1_AXP_s003
                                 + 0.970563 w_k1_DIS_s003
                                 + 1.09204 w_k1_FDX_s003
                                 + 0.815594 w_k1_IBM_s003
                                 + 1.10164 w_k1_INTC_s003
                                 + 0.995761 w_k1_KO_s003
                                 + 0.986915 w_k1_MAT_s003
                                 + 0.918217 w_k1_MSFT_s003
                                 + 1.0876 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.853399 w_k1_ADM_s003
                                 + 1.00043 w_k1_AXP_s003
                                 + 0.799004 w_k1_DIS_s003
                                 + 0.948198 w_k1_FDX_s003
                                 + 0.791975 w_k1_IBM_s003
                                 + 0.875346 w_k1_INTC_s003
                                 + 0.886281 w_k1_KO_s003
                                 + 1.46465 w_k1_MAT_s003
                                 + 0.896157 w_k1_MSFT_s003
                                 + 0.842707 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.826749 w_k1_ADM_s003
                                 + 0.856605 w_k1_AXP_s003
                                 + 0.947277 w_k1_DIS_s003
                                 + 0.844965 w_k1_FDX_s003
                                 + 1.28039 w_k1_IBM_s003
                                 + 1.0348 w_k1_INTC_s003 + 1.05372 w_k1_KO_s003
                                 + 1.54605 w_k1_MAT_s003
                                 + 0.980613 w_k1_MSFT_s003
                                 + 1.02415 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.1149 w_k1_ADM_s004
                                 + 1.07064 w_k1_AXP_s004
                                 + 1.02567 w_k1_DIS_s004
                                 + 1.05427 w_k1_FDX_s004
                                 + 1.24457 w_k1_IBM_s004
                                 + 1.10627 w_k1_INTC_s004
                                 + 1.33805 w_k1_KO_s004 + 1.34346 w_k1_MAT_s004
                                 + 0.993445 w_k1_MSFT_s004
                                 + 1.07534 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.09565 w_k1_ADM_s004
                                 + 0.665496 w_k1_AXP_s004
                                 + 0.742848 w_k1_DIS_s004
                                 + 0.870313 w_k1_FDX_s004
                                 + 0.663328 w_k1_IBM_s004
                                 + 0.700235 w_k1_INTC_s004
                                 + 0.939442 w_k1_KO_s004
                                 + 0.568041 w_k1_MAT_s004
                                 + 0.881417 w_k1_MSFT_s004
                                 + 0.832013 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.827758 w_k1_ADM_s004
                                 + 0.769213 w_k1_AXP_s004
                                 + 0.742694 w_k1_DIS_s004
                                 + 0.829664 w_k1_FDX_s004
                                 + 0.932933 w_k1_IBM_s004
                                 + 0.89515 w_k1_INTC_s004
                                 + 0.889491 w_k1_KO_s004
                                 + 0.838772 w_k1_MAT_s004
                                 + 0.853478 w_k1_MSFT_s004
                                 + 0.87376 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.59495 w_k1_ADM_s004
                                 + 1.63679 w_k1_AXP_s004
                                 + 1.55116 w_k1_DIS_s004
                                 + 1.36329 w_k1_FDX_s004
                                 + 1.21725 w_k1_IBM_s004
                                 + 1.08405 w_k1_INTC_s004
                                 + 1.41742 w_k1_KO_s004 + 1.40973 w_k1_MAT_s004
                                 + 1.08036 w_k1_MSFT_s004
                                 + 1.20212 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.25801 w_k1_ADM_s004
                                 + 0.988154 w_k1_AXP_s004
                                 + 1.0478 w_k1_DIS_s004 + 1.10354 w_k1_FDX_s004
                                 + 0.761319 w_k1_IBM_s004
                                 + 0.904017 w_k1_INTC_s004
                                 + 1.08304 w_k1_KO_s004
                                 + 0.801829 w_k1_MAT_s004
                                 + 1.07679 w_k1_MSFT_s004
                                 + 1.01529 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.14566 w_k1_ADM_s004
                                 + 1.33691 w_k1_AXP_s004
                                 + 1.15045 w_k1_DIS_s004
                                 + 1.15238 w_k1_FDX_s004
                                 + 1.63859 w_k1_IBM_s004
                                 + 1.06469 w_k1_INTC_s004
                                 + 1.52242 w_k1_KO_s004 + 1.97645 w_k1_MAT_s004
                                 + 1.15412 w_k1_MSFT_s004
                                 + 1.25921 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.246 w_k1_ADM_s004
                                 + 1.09963 w_k1_AXP_s004
                                 + 1.30898 w_k1_DIS_s004
                                 + 1.26398 w_k1_FDX_s004
                                 + 0.97588 w_k1_IBM_s004
                                 + 0.952614 w_k1_INTC_s004
                                 + 1.34229 w_k1_KO_s004
                                 + 0.824462 w_k1_MAT_s004
                                 + 1.22476 w_k1_MSFT_s004
                                 + 1.16151 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.21284 w_k1_ADM_s004
                                 + 1.11301 w_k1_AXP_s004
                                 + 0.872593 w_k1_DIS_s004
                                 + 0.919014 w_k1_FDX_s004
                                 + 1.08213 w_k1_IBM_s004
                                 + 0.843159 w_k1_INTC_s004
                                 + 1.07696 w_k1_KO_s004
                                 + 0.959416 w_k1_MAT_s004
                                 + 0.979503 w_k1_MSFT_s004
                                 + 0.95453 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.892202 w_k1_ADM_s005
                                 + 0.776917 w_k1_AXP_s005
                                 + 0.889749 w_k1_DIS_s005
                                 + 0.837246 w_k1_FDX_s005
                                 + 0.877285 w_k1_IBM_s005
                                 + 0.898107 w_k1_INTC_s005
                                 + 0.889777 w_k1_KO_s005
                                 + 0.600488 w_k1_MAT_s005
                                 + 0.887994 w_k1_MSFT_s005
                                 + 0.918863 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.16881 w_k1_ADM_s005
                                 + 1.17766 w_k1_AXP_s005
                                 + 1.01038 w_k1_DIS_s005
                                 + 1.05532 w_k1_FDX_s005
                                 + 1.14984 w_k1_IBM_s005
                                 + 0.88837 w_k1_INTC_s005
                                 + 1.04274 w_k1_KO_s005 + 1.36079 w_k1_MAT_s005
                                 + 1.08085 w_k1_MSFT_s005
                                 + 1.02898 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.1134 w_k1_ADM_s005
                                 + 1.22757 w_k1_AXP_s005
                                 + 1.00166 w_k1_DIS_s005
                                 + 0.947635 w_k1_FDX_s005
                                 + 1.61371 w_k1_IBM_s005
                                 + 1.06734 w_k1_INTC_s005
                                 + 1.17532 w_k1_KO_s005 + 1.4506 w_k1_MAT_s005
                                 + 0.89601 w_k1_MSFT_s005
                                 + 1.14679 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.656414 w_k1_ADM_s005
                                 + 0.696006 w_k1_AXP_s005
                                 + 0.664247 w_k1_DIS_s005
                                 + 0.681685 w_k1_FDX_s005
                                 + 0.846645 w_k1_IBM_s005
                                 + 0.761028 w_k1_INTC_s005
                                 + 0.735186 w_k1_KO_s005
                                 + 0.771295 w_k1_MAT_s005
                                 + 0.754434 w_k1_MSFT_s005
                                 + 0.849645 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.04167 w_k1_ADM_s005
                                 + 0.932175 w_k1_AXP_s005
                                 + 0.798307 w_k1_DIS_s005
                                 + 1.01751 w_k1_FDX_s005
                                 + 0.796464 w_k1_IBM_s005
                                 + 1.07474 w_k1_INTC_s005
                                 + 0.927046 w_k1_KO_s005
                                 + 0.783339 w_k1_MAT_s005
                                 + 1.03197 w_k1_MSFT_s005
                                 + 1.02861 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.10676 w_k1_ADM_s005
                                 + 1.10973 w_k1_AXP_s005
                                 + 1.38067 w_k1_DIS_s005
                                 + 1.23158 w_k1_FDX_s005
                                 + 1.00699 w_k1_IBM_s005
                                 + 1.15773 w_k1_INTC_s005
                                 + 1.13777 w_k1_KO_s005
                                 + 0.959136 w_k1_MAT_s005
                                 + 1.28288 w_k1_MSFT_s005
                                 + 1.10075 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.09174 w_k1_ADM_s005
                                 + 1.71671 w_k1_AXP_s005
                                 + 1.75487 w_k1_DIS_s005
                                 + 1.48816 w_k1_FDX_s005 + 1.4092 w_k1_IBM_s005
                                 + 1.47579 w_k1_INTC_s005
                                 + 1.24533 w_k1_KO_s005 + 1.81674 w_k1_MAT_s005
                                 + 1.61989 w_k1_MSFT_s005
                                 + 1.04672 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04564 w_k1_ADM_s005
                                 + 0.996174 w_k1_AXP_s005
                                 + 0.944118 w_k1_DIS_s005
                                 + 0.97609 w_k1_FDX_s005
                                 + 0.955727 w_k1_IBM_s005
                                 + 1.00232 w_k1_INTC_s005
                                 + 0.956882 w_k1_KO_s005
                                 + 0.816809 w_k1_MAT_s005
                                 + 1.03497 w_k1_MSFT_s005
                                 + 1.01043 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.906585 w_k1_ADM_s006
                                 + 1.09702 w_k1_AXP_s006
                                 + 1.42138 w_k1_DIS_s006
                                 + 1.07103 w_k1_FDX_s006
                                 + 1.60781 w_k1_IBM_s006
                                 + 1.00474 w_k1_INTC_s006 + 1.4694 w_k1_KO_s006
                                 + 2.25138 w_k1_MAT_s006
                                 + 1.17715 w_k1_MSFT_s006
                                 + 1.05669 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.796266 w_k1_ADM_s006
                                 + 0.53736 w_k1_AXP_s006
                                 + 0.629065 w_k1_DIS_s006
                                 + 0.701257 w_k1_FDX_s006
                                 + 0.73007 w_k1_IBM_s006
                                 + 0.74356 w_k1_INTC_s006
                                 + 0.806777 w_k1_KO_s006
                                 + 0.736754 w_k1_MAT_s006
                                 + 0.878299 w_k1_MSFT_s006
                                 + 0.794762 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.47922 w_k1_ADM_s006
                                 + 1.25717 w_k1_AXP_s006
                                 + 1.37271 w_k1_DIS_s006
                                 + 1.29894 w_k1_FDX_s006
                                 + 0.830661 w_k1_IBM_s006
                                 + 1.32035 w_k1_INTC_s006
                                 + 1.05547 w_k1_KO_s006 + 1.09249 w_k1_MAT_s006
                                 + 1.28273 w_k1_MSFT_s006
                                 + 1.03142 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.798563 w_k1_ADM_s006
                                 + 0.88378 w_k1_AXP_s006
                                 + 0.811087 w_k1_DIS_s006
                                 + 0.76325 w_k1_FDX_s006
                                 + 1.46311 w_k1_IBM_s006
                                 + 1.04571 w_k1_INTC_s006
                                 + 1.15943 w_k1_KO_s006 + 1.73971 w_k1_MAT_s006
                                 + 0.882791 w_k1_MSFT_s006
                                 + 0.93842 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.0463 w_k1_ADM_s006
                                 + 1.39299 w_k1_AXP_s006 + 1.1202 w_k1_DIS_s006
                                 + 1.063 w_k1_FDX_s006 + 1.28898 w_k1_IBM_s006
                                 + 0.94052 w_k1_INTC_s006
                                 + 1.15099 w_k1_KO_s006 + 1.56279 w_k1_MAT_s006
                                 + 0.940999 w_k1_MSFT_s006
                                 + 0.970948 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.09566 w_k1_ADM_s006
                                 + 1.05021 w_k1_AXP_s006
                                 + 0.974438 w_k1_DIS_s006
                                 + 1.01682 w_k1_FDX_s006
                                 + 0.848453 w_k1_IBM_s006
                                 + 1.00148 w_k1_INTC_s006
                                 + 0.844253 w_k1_KO_s006
                                 + 0.855801 w_k1_MAT_s006
                                 + 0.980781 w_k1_MSFT_s006
                                 + 0.863348 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.59674 w_k1_ADM_s006
                                 + 1.54638 w_k1_AXP_s006
                                 + 1.71301 w_k1_DIS_s006
                                 + 1.43756 w_k1_FDX_s006
                                 + 1.09896 w_k1_IBM_s006
                                 + 1.22951 w_k1_INTC_s006
                                 + 1.40742 w_k1_KO_s006 + 1.44089 w_k1_MAT_s006
                                 + 1.35807 w_k1_MSFT_s006 + 1.204 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.84625 w_k1_ADM_s006
                                 + 0.853448 w_k1_AXP_s006
                                 + 0.704805 w_k1_DIS_s006
                                 + 0.749454 w_k1_FDX_s006
                                 + 1.06758 w_k1_IBM_s006
                                 + 0.760077 w_k1_INTC_s006
                                 + 0.883271 w_k1_KO_s006
                                 + 1.05629 w_k1_MAT_s006
                                 + 0.891022 w_k1_MSFT_s006
                                 + 0.80195 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.00874 w_k1_ADM_s007
                                 + 0.836278 w_k1_AXP_s007
                                 + 0.712355 w_k1_DIS_s007
                                 + 0.879714 w_k1_FDX_s007
                                 + 0.562578 w_k1_IBM_s007
                                 + 1.14426 w_k1_INTC_s007
                                 + 0.822343 w_k1_KO_s007
                                 + 0.736128 w_k1_MAT_s007
                                 + 0.900077 w_k1_MSFT_s007
                                 + 0.905647 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.077 w_k1_ADM_s007
                                 + 1.03212 w_k1_AXP_s007
                                 + 0.980901 w_k1_DIS_s007
                                 + 0.998463 w_k1_FDX_s007
                                 + 1.04163 w_k1_IBM_s007
                                 + 0.834132 w_k1_INTC_s007
                                 + 1.22655 w_k1_KO_s007 + 1.52181 w_k1_MAT_s007
                                 + 0.904684 w_k1_MSFT_s007
                                 + 1.01462 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.886956 w_k1_ADM_s007
                                 + 0.634203 w_k1_AXP_s007
                                 + 0.613989 w_k1_DIS_s007
                                 + 0.726287 w_k1_FDX_s007
                                 + 0.797397 w_k1_IBM_s007
                                 + 0.84063 w_k1_INTC_s007
                                 + 0.80724 w_k1_KO_s007
                                 + 0.980157 w_k1_MAT_s007
                                 + 0.698014 w_k1_MSFT_s007
                                 + 0.861546 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.802507 w_k1_ADM_s007
                                 + 0.927705 w_k1_AXP_s007
                                 + 0.72952 w_k1_DIS_s007
                                 + 0.843889 w_k1_FDX_s007
                                 + 0.906034 w_k1_IBM_s007
                                 + 0.919554 w_k1_INTC_s007
                                 + 1.02455 w_k1_KO_s007 + 1.47048 w_k1_MAT_s007
                                 + 0.776927 w_k1_MSFT_s007
                                 + 0.991945 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02375 w_k1_ADM_s007
                                 + 1.11785 w_k1_AXP_s007
                                 + 0.840775 w_k1_DIS_s007
                                 + 1.04406 w_k1_FDX_s007
                                 + 0.80835 w_k1_IBM_s007
                                 + 1.06117 w_k1_INTC_s007
                                 + 0.865136 w_k1_KO_s007
                                 + 1.09763 w_k1_MAT_s007
                                 + 0.852311 w_k1_MSFT_s007
                                 + 0.953901 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.23307 w_k1_ADM_s007
                                 + 1.54496 w_k1_AXP_s007
                                 + 1.62052 w_k1_DIS_s007
                                 + 1.33004 w_k1_FDX_s007
                                 + 1.08703 w_k1_IBM_s007
                                 + 1.27676 w_k1_INTC_s007
                                 + 1.42086 w_k1_KO_s007 + 2.22504 w_k1_MAT_s007
                                 + 1.20537 w_k1_MSFT_s007
                                 + 1.19949 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.05171 w_k1_ADM_s007
                                 + 0.856196 w_k1_AXP_s007
                                 + 1.28684 w_k1_DIS_s007
                                 + 1.11169 w_k1_FDX_s007
                                 + 0.952572 w_k1_IBM_s007
                                 + 1.07694 w_k1_INTC_s007
                                 + 1.23029 w_k1_KO_s007
                                 + 0.954424 w_k1_MAT_s007
                                 + 1.1518 w_k1_MSFT_s007 + 1.11058 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.86764 w_k1_ADM_s007
                                 + 1.00444 w_k1_AXP_s007
                                 + 0.904966 w_k1_DIS_s007
                                 + 0.824709 w_k1_FDX_s007
                                 + 1.29523 w_k1_IBM_s007
                                 + 0.942453 w_k1_INTC_s007
                                 + 1.20599 w_k1_KO_s007 + 1.74422 w_k1_MAT_s007
                                 + 0.901124 w_k1_MSFT_s007
                                 + 1.07954 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.914642 w_k1_ADM_s010
                                 + 1.11504 w_k1_AXP_s010
                                 + 0.97366 w_k1_DIS_s010
                                 + 0.921764 w_k1_FDX_s010
                                 + 1.04361 w_k1_IBM_s010
                                 + 1.11129 w_k1_INTC_s010
                                 + 1.20714 w_k1_KO_s010
                                 + 0.995697 w_k1_MAT_s010
                                 + 0.977074 w_k1_MSFT_s010
                                 + 0.905304 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.77402 w_k1_ADM_s010
                                 + 0.65928 w_k1_AXP_s010
                                 + 0.734848 w_k1_DIS_s010
                                 + 0.833071 w_k1_FDX_s010
                                 + 0.758367 w_k1_IBM_s010
                                 + 0.886348 w_k1_INTC_s010
                                 + 0.850613 w_k1_KO_s010
                                 + 0.690616 w_k1_MAT_s010
                                 + 0.860891 w_k1_MSFT_s010
                                 + 0.898635 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.06582 w_k1_ADM_s010
                                 + 1.33849 w_k1_AXP_s010
                                 + 1.68106 w_k1_DIS_s010
                                 + 1.28244 w_k1_FDX_s010
                                 + 1.23412 w_k1_IBM_s010
                                 + 1.3953 w_k1_INTC_s010 + 1.41379 w_k1_KO_s010
                                 + 1.0833 w_k1_MAT_s010
                                 + 1.25769 w_k1_MSFT_s010
                                 + 1.15567 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.626589 w_k1_ADM_s010
                                 + 0.718237 w_k1_AXP_s010
                                 + 0.728675 w_k1_DIS_s010
                                 + 0.625153 w_k1_FDX_s010
                                 + 1.12362 w_k1_IBM_s010
                                 + 0.86258 w_k1_INTC_s010
                                 + 0.943562 w_k1_KO_s010
                                 + 1.16314 w_k1_MAT_s010
                                 + 0.834499 w_k1_MSFT_s010
                                 + 0.770369 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.888773 w_k1_ADM_s010
                                 + 1.13678 w_k1_AXP_s010
                                 + 1.17715 w_k1_DIS_s010
                                 + 1.10281 w_k1_FDX_s010
                                 + 0.832171 w_k1_IBM_s010
                                 + 1.27664 w_k1_INTC_s010 + 1.0483 w_k1_KO_s010
                                 + 1.05908 w_k1_MAT_s010
                                 + 1.23733 w_k1_MSFT_s010
                                 + 0.881649 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.23641 w_k1_ADM_s010
                                 + 1.63908 w_k1_AXP_s010
                                 + 1.65703 w_k1_DIS_s010
                                 + 1.26937 w_k1_FDX_s010
                                 + 1.11065 w_k1_IBM_s010
                                 + 1.60767 w_k1_INTC_s010
                                 + 1.43374 w_k1_KO_s010 + 1.51645 w_k1_MAT_s010
                                 + 1.29833 w_k1_MSFT_s010
                                 + 1.18498 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.685584 w_k1_ADM_s010
                                 + 1.3693 w_k1_AXP_s010 + 1.14267 w_k1_DIS_s010
                                 + 1.02627 w_k1_FDX_s010
                                 + 1.40205 w_k1_IBM_s010
                                 + 1.44484 w_k1_INTC_s010
                                 + 1.23104 w_k1_KO_s010 + 1.89022 w_k1_MAT_s010
                                 + 1.06748 w_k1_MSFT_s010
                                 + 1.12299 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.901902 w_k1_ADM_s010
                                 + 1.45858 w_k1_AXP_s010 + 1.1873 w_k1_DIS_s010
                                 + 0.990789 w_k1_FDX_s010
                                 + 1.77365 w_k1_IBM_s010
                                 + 1.19797 w_k1_INTC_s010
                                 + 1.52368 w_k1_KO_s010 + 2.1007 w_k1_MAT_s010
                                 + 1.0642 w_k1_MSFT_s010 + 1.1022 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.9709 w_k1_ADM_s011
                                 + 1.1566 w_k1_AXP_s011
                                 + 0.942907 w_k1_DIS_s011
                                 + 1.02025 w_k1_FDX_s011
                                 + 1.38983 w_k1_IBM_s011
                                 + 1.09932 w_k1_INTC_s011
                                 + 1.09374 w_k1_KO_s011 + 1.73657 w_k1_MAT_s011
                                 + 1.02866 w_k1_MSFT_s011 + 1.1076 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.953174 w_k1_ADM_s011
                                 + 0.908246 w_k1_AXP_s011
                                 + 0.812405 w_k1_DIS_s011
                                 + 0.967082 w_k1_FDX_s011
                                 + 0.612274 w_k1_IBM_s011
                                 + 0.918094 w_k1_INTC_s011
                                 + 0.840675 w_k1_KO_s011
                                 + 0.530888 w_k1_MAT_s011
                                 + 1.06006 w_k1_MSFT_s011
                                 + 0.931402 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.957906 w_k1_ADM_s011
                                 + 1.07779 w_k1_AXP_s011
                                 + 0.817464 w_k1_DIS_s011
                                 + 1.00503 w_k1_FDX_s011
                                 + 0.860742 w_k1_IBM_s011
                                 + 0.805495 w_k1_INTC_s011
                                 + 0.852931 w_k1_KO_s011
                                 + 1.05831 w_k1_MAT_s011
                                 + 0.958923 w_k1_MSFT_s011
                                 + 0.958836 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.38336 w_k1_ADM_s011
                                 + 1.75713 w_k1_AXP_s011
                                 + 1.42687 w_k1_DIS_s011
                                 + 1.28607 w_k1_FDX_s011
                                 + 1.52643 w_k1_IBM_s011
                                 + 1.22135 w_k1_INTC_s011
                                 + 1.36267 w_k1_KO_s011 + 1.96514 w_k1_MAT_s011
                                 + 1.35603 w_k1_MSFT_s011
                                 + 1.15209 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.822099 w_k1_ADM_s011
                                 + 0.845085 w_k1_AXP_s011
                                 + 0.678675 w_k1_DIS_s011
                                 + 0.84467 w_k1_FDX_s011
                                 + 0.905973 w_k1_IBM_s011
                                 + 0.870521 w_k1_INTC_s011
                                 + 0.920129 w_k1_KO_s011
                                 + 0.720964 w_k1_MAT_s011
                                 + 0.941728 w_k1_MSFT_s011
                                 + 0.917305 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.26402 w_k1_ADM_s011
                                 + 1.2886 w_k1_AXP_s011 + 1.1585 w_k1_DIS_s011
                                 + 1.28725 w_k1_FDX_s011
                                 + 0.940701 w_k1_IBM_s011
                                 + 1.13979 w_k1_INTC_s011
                                 + 1.11164 w_k1_KO_s011 + 1.02747 w_k1_MAT_s011
                                 + 1.2429 w_k1_MSFT_s011
                                 + 0.989432 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06124 w_k1_ADM_s011
                                 + 0.787146 w_k1_AXP_s011
                                 + 0.857162 w_k1_DIS_s011
                                 + 0.926938 w_k1_FDX_s011
                                 + 1.16128 w_k1_IBM_s011
                                 + 0.997902 w_k1_INTC_s011
                                 + 1.21551 w_k1_KO_s011 + 1.01837 w_k1_MAT_s011
                                 + 1.06868 w_k1_MSFT_s011 + 1.0215 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.913412 w_k1_ADM_s011
                                 + 0.508562 w_k1_AXP_s011
                                 + 0.610402 w_k1_DIS_s011
                                 + 0.681726 w_k1_FDX_s011
                                 + 0.714125 w_k1_IBM_s011
                                 + 0.666384 w_k1_INTC_s011
                                 + 0.765324 w_k1_KO_s011
                                 + 0.605751 w_k1_MAT_s011
                                 + 0.858145 w_k1_MSFT_s011
                                 + 0.738699 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.15687 w_k1_ADM_s012
                                 + 0.883993 w_k1_AXP_s012
                                 + 1.14188 w_k1_DIS_s012
                                 + 1.07143 w_k1_FDX_s012
                                 + 0.873545 w_k1_IBM_s012
                                 + 1.08188 w_k1_INTC_s012
                                 + 1.02307 w_k1_KO_s012
                                 + 0.911475 w_k1_MAT_s012
                                 + 1.1294 w_k1_MSFT_s012
                                 + 0.972766 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.830982 w_k1_ADM_s012
                                 + 0.892551 w_k1_AXP_s012
                                 + 0.896146 w_k1_DIS_s012
                                 + 0.815719 w_k1_FDX_s012
                                 + 1.17947 w_k1_IBM_s012
                                 + 0.934157 w_k1_INTC_s012
                                 + 0.974301 w_k1_KO_s012
                                 + 1.08511 w_k1_MAT_s012
                                 + 0.962256 w_k1_MSFT_s012
                                 + 0.92838 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.03316 w_k1_ADM_s012
                                 + 0.929432 w_k1_AXP_s012
                                 + 1.14892 w_k1_DIS_s012
                                 + 0.946607 w_k1_FDX_s012
                                 + 1.60317 w_k1_IBM_s012
                                 + 0.95811 w_k1_INTC_s012
                                 + 1.23384 w_k1_KO_s012 + 1.25527 w_k1_MAT_s012
                                 + 0.923426 w_k1_MSFT_s012
                                 + 0.924394 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.883569 w_k1_ADM_s012
                                 + 1.20463 w_k1_AXP_s012
                                 + 0.960517 w_k1_DIS_s012
                                 + 0.987355 w_k1_FDX_s012
                                 + 1.11447 w_k1_IBM_s012
                                 + 1.12788 w_k1_INTC_s012
                                 + 0.988051 w_k1_KO_s012
                                 + 1.79962 w_k1_MAT_s012
                                 + 0.96508 w_k1_MSFT_s012
                                 + 0.957672 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.23621 w_k1_ADM_s012
                                 + 1.2146 w_k1_AXP_s012 + 1.68356 w_k1_DIS_s012
                                 + 1.38409 w_k1_FDX_s012
                                 + 1.16271 w_k1_IBM_s012
                                 + 1.28365 w_k1_INTC_s012
                                 + 1.33967 w_k1_KO_s012 + 1.4572 w_k1_MAT_s012
                                 + 1.44702 w_k1_MSFT_s012
                                 + 1.15251 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.20976 w_k1_ADM_s012
                                 + 1.40463 w_k1_AXP_s012
                                 + 1.05456 w_k1_DIS_s012
                                 + 1.14678 w_k1_FDX_s012 + 1.1052 w_k1_IBM_s012
                                 + 1.13811 w_k1_INTC_s012 + 0.8766 w_k1_KO_s012
                                 + 1.17794 w_k1_MAT_s012
                                 + 0.998049 w_k1_MSFT_s012
                                 + 1.0947 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.18949 w_k1_ADM_s012
                                 + 1.45314 w_k1_AXP_s012
                                 + 1.57993 w_k1_DIS_s012
                                 + 1.15694 w_k1_FDX_s012
                                 + 1.71069 w_k1_IBM_s012
                                 + 1.2544 w_k1_INTC_s012 + 1.58591 w_k1_KO_s012
                                 + 1.48722 w_k1_MAT_s012
                                 + 1.14693 w_k1_MSFT_s012
                                 + 1.15951 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.962924 w_k1_ADM_s012
                                 + 0.720164 w_k1_AXP_s012
                                 + 0.66489 w_k1_DIS_s012
                                 + 0.838662 w_k1_FDX_s012
                                 + 0.752445 w_k1_IBM_s012
                                 + 0.804227 w_k1_INTC_s012
                                 + 0.741073 w_k1_KO_s012
                                 + 0.800382 w_k1_MAT_s012
                                 + 0.83899 w_k1_MSFT_s012
                                 + 0.76348 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.48355 w_k1_ADM_s013
                                 + 1.97343 w_k1_AXP_s013
                                 + 1.97749 w_k1_DIS_s013
                                 + 1.66937 w_k1_FDX_s013
                                 + 1.24029 w_k1_IBM_s013
                                 + 1.51981 w_k1_INTC_s013
                                 + 1.24519 w_k1_KO_s013 + 1.17827 w_k1_MAT_s013
                                 + 1.44113 w_k1_MSFT_s013
                                 + 1.45263 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.04484 w_k1_ADM_s013
                                 + 1.07276 w_k1_AXP_s013
                                 + 0.92707 w_k1_DIS_s013
                                 + 0.896024 w_k1_FDX_s013
                                 + 1.48127 w_k1_IBM_s013
                                 + 1.03642 w_k1_INTC_s013
                                 + 1.11369 w_k1_KO_s013 + 1.20444 w_k1_MAT_s013
                                 + 1.05306 w_k1_MSFT_s013
                                 + 1.00214 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.19481 w_k1_ADM_s013
                                 + 1.1015 w_k1_AXP_s013 + 0.92266 w_k1_DIS_s013
                                 + 1.04048 w_k1_FDX_s013
                                 + 0.65453 w_k1_IBM_s013
                                 + 1.27094 w_k1_INTC_s013
                                 + 0.763385 w_k1_KO_s013
                                 + 0.543594 w_k1_MAT_s013
                                 + 1.02941 w_k1_MSFT_s013
                                 + 0.953003 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.35433 w_k1_ADM_s013
                                 + 1.10642 w_k1_AXP_s013
                                 + 1.58419 w_k1_DIS_s013
                                 + 1.36922 w_k1_FDX_s013
                                 + 1.06759 w_k1_IBM_s013
                                 + 1.54002 w_k1_INTC_s013
                                 + 1.24065 w_k1_KO_s013 + 0.65213 w_k1_MAT_s013
                                 + 1.33774 w_k1_MSFT_s013
                                 + 1.15296 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.943314 w_k1_ADM_s013
                                 + 1.39676 w_k1_AXP_s013
                                 + 1.29141 w_k1_DIS_s013
                                 + 1.15124 w_k1_FDX_s013
                                 + 1.39157 w_k1_IBM_s013
                                 + 1.61689 w_k1_INTC_s013
                                 + 1.11801 w_k1_KO_s013 + 1.40516 w_k1_MAT_s013
                                 + 1.23669 w_k1_MSFT_s013
                                 + 1.12923 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.14122 w_k1_ADM_s013
                                 + 1.16019 w_k1_AXP_s013
                                 + 1.19974 w_k1_DIS_s013
                                 + 1.16957 w_k1_FDX_s013
                                 + 0.854303 w_k1_IBM_s013
                                 + 1.16996 w_k1_INTC_s013
                                 + 0.895829 w_k1_KO_s013
                                 + 0.546956 w_k1_MAT_s013
                                 + 1.10117 w_k1_MSFT_s013
                                 + 1.05141 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.54436 w_k1_ADM_s013
                                 + 0.64015 w_k1_AXP_s013
                                 + 0.607046 w_k1_DIS_s013
                                 + 0.69416 w_k1_FDX_s013
                                 + 0.924774 w_k1_IBM_s013
                                 + 0.849588 w_k1_INTC_s013
                                 + 0.673065 w_k1_KO_s013
                                 + 0.709541 w_k1_MAT_s013
                                 + 0.702592 w_k1_MSFT_s013
                                 + 0.767624 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.96957 w_k1_ADM_s013
                                 + 0.863853 w_k1_AXP_s013
                                 + 0.741897 w_k1_DIS_s013
                                 + 0.885823 w_k1_FDX_s013
                                 + 0.859164 w_k1_IBM_s013
                                 + 1.09371 w_k1_INTC_s013
                                 + 0.810662 w_k1_KO_s013
                                 + 0.656463 w_k1_MAT_s013
                                 + 1.01395 w_k1_MSFT_s013
                                 + 0.935819 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.23319 w_k1_ADM_s014
                                 + 1.46438 w_k1_AXP_s014
                                 + 1.53972 w_k1_DIS_s014
                                 + 1.23706 w_k1_FDX_s014
                                 + 1.57102 w_k1_IBM_s014
                                 + 1.1093 w_k1_INTC_s014 + 1.48828 w_k1_KO_s014
                                 + 2.10366 w_k1_MAT_s014
                                 + 1.21933 w_k1_MSFT_s014
                                 + 1.25107 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.32293 w_k1_ADM_s014
                                 + 1.23469 w_k1_AXP_s014
                                 + 1.29949 w_k1_DIS_s014
                                 + 1.28363 w_k1_FDX_s014 + w_k1_IBM_s014
                                 + 1.32118 w_k1_INTC_s014
                                 + 1.15272 w_k1_KO_s014 + 1.06471 w_k1_MAT_s014
                                 + 1.16849 w_k1_MSFT_s014
                                 + 1.00644 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.30994 w_k1_ADM_s014
                                 + 0.976072 w_k1_AXP_s014
                                 + 1.10035 w_k1_DIS_s014
                                 + 1.06146 w_k1_FDX_s014
                                 + 0.82914 w_k1_IBM_s014
                                 + 1.00852 w_k1_INTC_s014
                                 + 1.10286 w_k1_KO_s014 + 1.1683 w_k1_MAT_s014
                                 + 1.06912 w_k1_MSFT_s014
                                 + 1.08867 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.897984 w_k1_ADM_s014
                                 + 0.675578 w_k1_AXP_s014
                                 + 0.629826 w_k1_DIS_s014
                                 + 0.679351 w_k1_FDX_s014
                                 + 0.895218 w_k1_IBM_s014
                                 + 0.863713 w_k1_INTC_s014
                                 + 0.84594 w_k1_KO_s014
                                 + 0.876197 w_k1_MAT_s014
                                 + 0.776386 w_k1_MSFT_s014
                                 + 0.815246 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.881911 w_k1_ADM_s014
                                 + 0.962363 w_k1_AXP_s014
                                 + 0.767979 w_k1_DIS_s014
                                 + 0.862431 w_k1_FDX_s014
                                 + 0.880524 w_k1_IBM_s014
                                 + 0.988303 w_k1_INTC_s014
                                 + 0.703361 w_k1_KO_s014
                                 + 1.03719 w_k1_MAT_s014
                                 + 0.88543 w_k1_MSFT_s014
                                 + 0.908896 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.953148 w_k1_ADM_s014
                                 + 0.88401 w_k1_AXP_s014
                                 + 1.05725 w_k1_DIS_s014
                                 + 0.88839 w_k1_FDX_s014
                                 + 1.23725 w_k1_IBM_s014
                                 + 0.935181 w_k1_INTC_s014
                                 + 1.14052 w_k1_KO_s014 + 1.27466 w_k1_MAT_s014
                                 + 1.01816 w_k1_MSFT_s014
                                 + 0.96795 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.977466 w_k1_ADM_s014
                                 + 0.836243 w_k1_AXP_s014
                                 + 0.887792 w_k1_DIS_s014
                                 + 0.998153 w_k1_FDX_s014
                                 + 0.708486 w_k1_IBM_s014
                                 + 0.904808 w_k1_INTC_s014
                                 + 0.80432 w_k1_KO_s014 + 1.09225 w_k1_MAT_s014
                                 + 1.10098 w_k1_MSFT_s014
                                 + 0.925547 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.04091 w_k1_ADM_s014
                                 + 1.22425 w_k1_AXP_s014
                                 + 0.917671 w_k1_DIS_s014
                                 + 1.06234 w_k1_FDX_s014
                                 + 1.06384 w_k1_IBM_s014
                                 + 1.05399 w_k1_INTC_s014
                                 + 1.07606 w_k1_KO_s014 + 1.38826 w_k1_MAT_s014
                                 + 0.972645 w_k1_MSFT_s014
                                 + 1.06279 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.15109 w_k1_ADM_s015
                                 + 1.09348 w_k1_AXP_s015
                                 + 1.15956 w_k1_DIS_s015
                                 + 1.06267 w_k1_FDX_s015
                                 + 1.23802 w_k1_IBM_s015
                                 + 0.984544 w_k1_INTC_s015
                                 + 1.20257 w_k1_KO_s015 + 1.35421 w_k1_MAT_s015
                                 + 1.1794 w_k1_MSFT_s015 + 1.14469 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.786497 w_k1_ADM_s015
                                 + 0.846703 w_k1_AXP_s015
                                 + 0.98925 w_k1_DIS_s015
                                 + 0.886773 w_k1_FDX_s015
                                 + 0.939019 w_k1_IBM_s015
                                 + 0.803898 w_k1_INTC_s015
                                 + 0.951386 w_k1_KO_s015
                                 + 1.12823 w_k1_MAT_s015
                                 + 1.05222 w_k1_MSFT_s015
                                 + 0.939392 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.962532 w_k1_ADM_s015
                                 + 0.941936 w_k1_AXP_s015
                                 + 0.732903 w_k1_DIS_s015
                                 + 0.8046 w_k1_FDX_s015 + 1.0897 w_k1_IBM_s015
                                 + 0.789991 w_k1_INTC_s015
                                 + 0.981751 w_k1_KO_s015
                                 + 1.54564 w_k1_MAT_s015
                                 + 0.920139 w_k1_MSFT_s015
                                 + 0.921789 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.965399 w_k1_ADM_s015
                                 + 1.09786 w_k1_AXP_s015
                                 + 0.944236 w_k1_DIS_s015
                                 + 0.895493 w_k1_FDX_s015
                                 + 1.49675 w_k1_IBM_s015
                                 + 1.06467 w_k1_INTC_s015
                                 + 1.25239 w_k1_KO_s015 + 1.68991 w_k1_MAT_s015
                                 + 1.0367 w_k1_MSFT_s015 + 1.05467 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.791676 w_k1_ADM_s015
                                 + 0.75215 w_k1_AXP_s015
                                 + 0.646173 w_k1_DIS_s015
                                 + 0.790564 w_k1_FDX_s015
                                 + 0.595301 w_k1_IBM_s015
                                 + 0.762124 w_k1_INTC_s015
                                 + 0.648867 w_k1_KO_s015
                                 + 0.591072 w_k1_MAT_s015
                                 + 0.925036 w_k1_MSFT_s015
                                 + 0.781907 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.916527 w_k1_ADM_s015
                                 + 1.15488 w_k1_AXP_s015
                                 + 1.09092 w_k1_DIS_s015 + 1.1132 w_k1_FDX_s015
                                 + 0.80604 w_k1_IBM_s015
                                 + 0.929927 w_k1_INTC_s015
                                 + 0.946257 w_k1_KO_s015
                                 + 0.894557 w_k1_MAT_s015
                                 + 1.17035 w_k1_MSFT_s015
                                 + 1.09267 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.918359 w_k1_ADM_s015
                                 + 0.671554 w_k1_AXP_s015
                                 + 0.838269 w_k1_DIS_s015
                                 + 0.881994 w_k1_FDX_s015
                                 + 0.711651 w_k1_IBM_s015
                                 + 0.663867 w_k1_INTC_s015
                                 + 0.798632 w_k1_KO_s015
                                 + 0.812762 w_k1_MAT_s015
                                 + 1.00137 w_k1_MSFT_s015
                                 + 0.838852 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.35042 w_k1_ADM_s015
                                 + 1.40567 w_k1_AXP_s015
                                 + 1.62799 w_k1_DIS_s015
                                 + 1.27379 w_k1_FDX_s015
                                 + 1.27912 w_k1_IBM_s015
                                 + 0.916035 w_k1_INTC_s015
                                 + 1.43718 w_k1_KO_s015 + 1.56629 w_k1_MAT_s015
                                 + 1.20002 w_k1_MSFT_s015
                                 + 1.21605 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.763009 w_k1_ADM_s016
                                 + 1.05234 w_k1_AXP_s016
                                 + 1.00986 w_k1_DIS_s016
                                 + 0.880844 w_k1_FDX_s016
                                 + 1.48959 w_k1_IBM_s016
                                 + 1.06598 w_k1_INTC_s016
                                 + 1.32101 w_k1_KO_s016 + 1.81467 w_k1_MAT_s016
                                 + 1.00424 w_k1_MSFT_s016
                                 + 0.986275 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.941295 w_k1_ADM_s016
                                 + 1.03467 w_k1_AXP_s016
                                 + 1.06484 w_k1_DIS_s016
                                 + 0.907176 w_k1_FDX_s016
                                 + 1.17392 w_k1_IBM_s016
                                 + 1.06226 w_k1_INTC_s016
                                 + 1.11204 w_k1_KO_s016
                                 + 0.834572 w_k1_MAT_s016
                                 + 0.947812 w_k1_MSFT_s016
                                 + 0.982597 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.761182 w_k1_ADM_s016
                                 + 0.852441 w_k1_AXP_s016
                                 + 0.949107 w_k1_DIS_s016
                                 + 0.868034 w_k1_FDX_s016
                                 + 0.810088 w_k1_IBM_s016
                                 + 1.02182 w_k1_INTC_s016
                                 + 0.811161 w_k1_KO_s016
                                 + 0.730187 w_k1_MAT_s016
                                 + 1.01389 w_k1_MSFT_s016
                                 + 0.927206 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.967044 w_k1_ADM_s016
                                 + 1.52179 w_k1_AXP_s016
                                 + 1.80612 w_k1_DIS_s016
                                 + 1.30143 w_k1_FDX_s016
                                 + 1.50492 w_k1_IBM_s016
                                 + 1.46099 w_k1_INTC_s016 + 1.369 w_k1_KO_s016
                                 + 1.6622 w_k1_MAT_s016 + 1.2624 w_k1_MSFT_s016
                                 + 1.28866 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10055 w_k1_ADM_s016
                                 + 1.22952 w_k1_AXP_s016
                                 + 1.39686 w_k1_DIS_s016
                                 + 1.29594 w_k1_FDX_s016
                                 + 0.921914 w_k1_IBM_s016
                                 + 1.51407 w_k1_INTC_s016
                                 + 1.08995 w_k1_KO_s016
                                 + 0.824455 w_k1_MAT_s016
                                 + 1.32887 w_k1_MSFT_s016
                                 + 1.16858 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.04943 w_k1_ADM_s016
                                 + 0.967101 w_k1_AXP_s016
                                 + 1.12514 w_k1_DIS_s016
                                 + 1.10235 w_k1_FDX_s016
                                 + 0.628295 w_k1_IBM_s016
                                 + 0.98152 w_k1_INTC_s016
                                 + 0.802034 w_k1_KO_s016
                                 + 0.736034 w_k1_MAT_s016
                                 + 0.929818 w_k1_MSFT_s016
                                 + 0.927121 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.863198 w_k1_ADM_s016
                                 + 0.752497 w_k1_AXP_s016
                                 + 0.687879 w_k1_DIS_s016
                                 + 0.799096 w_k1_FDX_s016
                                 + 0.590973 w_k1_IBM_s016
                                 + 0.785491 w_k1_INTC_s016
                                 + 0.677662 w_k1_KO_s016
                                 + 0.481482 w_k1_MAT_s016
                                 + 0.741752 w_k1_MSFT_s016
                                 + 0.877099 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.653123 w_k1_ADM_s016
                                 + 0.850038 w_k1_AXP_s016
                                 + 0.714331 w_k1_DIS_s016
                                 + 0.768128 w_k1_FDX_s016
                                 + 0.977479 w_k1_IBM_s016
                                 + 0.962114 w_k1_INTC_s016
                                 + 0.810311 w_k1_KO_s016
                                 + 0.963077 w_k1_MAT_s016
                                 + 0.84733 w_k1_MSFT_s016
                                 + 0.941816 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.883063 w_k1_ADM_s017
                                 + 0.788831 w_k1_AXP_s017
                                 + 0.609657 w_k1_DIS_s017
                                 + 0.713475 w_k1_FDX_s017
                                 + 0.699099 w_k1_IBM_s017
                                 + 0.854951 w_k1_INTC_s017
                                 + 0.699852 w_k1_KO_s017
                                 + 0.731847 w_k1_MAT_s017
                                 + 0.760201 w_k1_MSFT_s017
                                 + 0.755456 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.758811 w_k1_ADM_s017
                                 + 0.822206 w_k1_AXP_s017
                                 + 0.815791 w_k1_DIS_s017
                                 + 0.840758 w_k1_FDX_s017
                                 + 1.07491 w_k1_IBM_s017
                                 + 1.02033 w_k1_INTC_s017
                                 + 0.932986 w_k1_KO_s017
                                 + 0.697706 w_k1_MAT_s017
                                 + 0.790936 w_k1_MSFT_s017
                                 + 0.878581 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.52802 w_k1_ADM_s017
                                 + 1.22802 w_k1_AXP_s017
                                 + 1.52343 w_k1_DIS_s017
                                 + 1.25012 w_k1_FDX_s017
                                 + 1.08266 w_k1_IBM_s017
                                 + 1.17999 w_k1_INTC_s017
                                 + 1.26843 w_k1_KO_s017
                                 + 0.891787 w_k1_MAT_s017
                                 + 1.3524 w_k1_MSFT_s017 + 1.15616 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.903225 w_k1_ADM_s017
                                 + 1.34621 w_k1_AXP_s017
                                 + 1.30925 w_k1_DIS_s017
                                 + 1.08631 w_k1_FDX_s017
                                 + 1.95597 w_k1_IBM_s017
                                 + 1.14199 w_k1_INTC_s017 + 1.2538 w_k1_KO_s017
                                 + 1.75557 w_k1_MAT_s017
                                 + 1.19495 w_k1_MSFT_s017
                                 + 1.14115 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0805 w_k1_ADM_s017
                                 + 0.947913 w_k1_AXP_s017
                                 + 1.07295 w_k1_DIS_s017
                                 + 1.13094 w_k1_FDX_s017
                                 + 0.706362 w_k1_IBM_s017
                                 + 1.13673 w_k1_INTC_s017
                                 + 0.882696 w_k1_KO_s017
                                 + 0.579614 w_k1_MAT_s017
                                 + 1.08236 w_k1_MSFT_s017
                                 + 0.88152 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.819256 w_k1_ADM_s017
                                 + 1.09114 w_k1_AXP_s017
                                 + 0.900159 w_k1_DIS_s017
                                 + 0.880178 w_k1_FDX_s017
                                 + 1.53065 w_k1_IBM_s017
                                 + 1.09591 w_k1_INTC_s017
                                 + 1.14681 w_k1_KO_s017 + 1.42663 w_k1_MAT_s017
                                 + 0.98435 w_k1_MSFT_s017
                                 + 0.936264 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.01755 w_k1_ADM_s017
                                 + 0.784486 w_k1_AXP_s017
                                 + 0.85474 w_k1_DIS_s017
                                 + 0.906258 w_k1_FDX_s017
                                 + 0.780375 w_k1_IBM_s017
                                 + 0.994584 w_k1_INTC_s017
                                 + 0.829609 w_k1_KO_s017
                                 + 0.560045 w_k1_MAT_s017
                                 + 1.01827 w_k1_MSFT_s017
                                 + 0.91911 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.03324 w_k1_ADM_s017
                                 + 1.30113 w_k1_AXP_s017
                                 + 1.16509 w_k1_DIS_s017
                                 + 1.09831 w_k1_FDX_s017
                                 + 1.20688 w_k1_IBM_s017
                                 + 0.976781 w_k1_INTC_s017
                                 + 1.09147 w_k1_KO_s017 + 1.31561 w_k1_MAT_s017
                                 + 1.02885 w_k1_MSFT_s017 + 1.0414 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.746419 w_k1_ADM_s100
                                 + 0.963119 w_k1_AXP_s100
                                 + 0.949647 w_k1_DIS_s100
                                 + 1.05777 w_k1_FDX_s100
                                 + 0.966401 w_k1_IBM_s100
                                 + 1.05004 w_k1_INTC_s100
                                 + 0.951727 w_k1_KO_s100
                                 + 0.880763 w_k1_MAT_s100
                                 + 0.992994 w_k1_MSFT_s100
                                 + 0.988973 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.974794 w_k1_ADM_s100
                                 + 0.876426 w_k1_AXP_s100
                                 + 0.739186 w_k1_DIS_s100
                                 + 0.969365 w_k1_FDX_s100
                                 + 0.656188 w_k1_IBM_s100
                                 + 0.726493 w_k1_INTC_s100
                                 + 0.772294 w_k1_KO_s100
                                 + 0.62313 w_k1_MAT_s100
                                 + 0.900689 w_k1_MSFT_s100
                                 + 0.951025 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.03162 w_k1_ADM_s100
                                 + 1.35478 w_k1_AXP_s100
                                 + 1.25733 w_k1_DIS_s100
                                 + 1.13951 w_k1_FDX_s100
                                 + 0.856977 w_k1_IBM_s100
                                 + 1.15009 w_k1_INTC_s100
                                 + 1.13089 w_k1_KO_s100 + 1.14031 w_k1_MAT_s100
                                 + 1.13301 w_k1_MSFT_s100
                                 + 1.02387 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.631884 w_k1_ADM_s100
                                 + 0.701306 w_k1_AXP_s100
                                 + 0.60371 w_k1_DIS_s100
                                 + 0.725968 w_k1_FDX_s100
                                 + 0.757592 w_k1_IBM_s100
                                 + 0.773076 w_k1_INTC_s100
                                 + 0.764971 w_k1_KO_s100
                                 + 0.96588 w_k1_MAT_s100
                                 + 0.729452 w_k1_MSFT_s100
                                 + 0.771544 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.12017 w_k1_ADM_s100
                                 + 1.37791 w_k1_AXP_s100
                                 + 1.18967 w_k1_DIS_s100
                                 + 1.09183 w_k1_FDX_s100
                                 + 1.34142 w_k1_IBM_s100
                                 + 1.13857 w_k1_INTC_s100
                                 + 1.28609 w_k1_KO_s100 + 1.34888 w_k1_MAT_s100
                                 + 1.02643 w_k1_MSFT_s100
                                 + 1.08019 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.592511 w_k1_ADM_s100
                                 + 0.820979 w_k1_AXP_s100
                                 + 1.00954 w_k1_DIS_s100
                                 + 0.864641 w_k1_FDX_s100
                                 + 1.14778 w_k1_IBM_s100
                                 + 0.894648 w_k1_INTC_s100
                                 + 1.04413 w_k1_KO_s100 + 1.1351 w_k1_MAT_s100
                                 + 0.955349 w_k1_MSFT_s100
                                 + 1.09306 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.622081 w_k1_ADM_s100
                                 + 0.897923 w_k1_AXP_s100
                                 + 0.739493 w_k1_DIS_s100
                                 + 0.776727 w_k1_FDX_s100
                                 + 0.978296 w_k1_IBM_s100
                                 + 0.807125 w_k1_INTC_s100
                                 + 0.938131 w_k1_KO_s100
                                 + 0.933071 w_k1_MAT_s100
                                 + 0.820155 w_k1_MSFT_s100
                                 + 0.873806 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.703048 w_k1_ADM_s100
                                 + 0.830055 w_k1_AXP_s100
                                 + 0.911617 w_k1_DIS_s100
                                 + 0.896559 w_k1_FDX_s100
                                 + 0.778676 w_k1_IBM_s100
                                 + 0.98386 w_k1_INTC_s100
                                 + 0.956415 w_k1_KO_s100
                                 + 0.543513 w_k1_MAT_s100
                                 + 0.996766 w_k1_MSFT_s100
                                 + 1.00549 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.20542 w_k1_ADM_s101
                                 + 1.65673 w_k1_AXP_s101
                                 + 1.70659 w_k1_DIS_s101
                                 + 1.40147 w_k1_FDX_s101
                                 + 1.08585 w_k1_IBM_s101
                                 + 1.27553 w_k1_INTC_s101
                                 + 1.37438 w_k1_KO_s101 + 1.72637 w_k1_MAT_s101
                                 + 1.48342 w_k1_MSFT_s101 + 1.2037 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.722726 w_k1_ADM_s101
                                 + 0.904933 w_k1_AXP_s101
                                 + 0.759387 w_k1_DIS_s101
                                 + 0.795498 w_k1_FDX_s101
                                 + 0.953186 w_k1_IBM_s101
                                 + 0.860075 w_k1_INTC_s101
                                 + 0.891394 w_k1_KO_s101 + 1.2359 w_k1_MAT_s101
                                 + 0.802523 w_k1_MSFT_s101
                                 + 1.01263 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.842675 w_k1_ADM_s101
                                 + 0.84506 w_k1_AXP_s101
                                 + 0.679442 w_k1_DIS_s101
                                 + 0.7063 w_k1_FDX_s101
                                 + 0.745297 w_k1_IBM_s101
                                 + 0.816527 w_k1_INTC_s101
                                 + 0.754275 w_k1_KO_s101
                                 + 0.805004 w_k1_MAT_s101
                                 + 0.727995 w_k1_MSFT_s101
                                 + 0.915015 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.00569 w_k1_ADM_s101
                                 + 1.2807 w_k1_AXP_s101 + 1.42706 w_k1_DIS_s101
                                 + 1.09072 w_k1_FDX_s101
                                 + 1.29872 w_k1_IBM_s101
                                 + 1.09055 w_k1_INTC_s101
                                 + 1.36729 w_k1_KO_s101 + 1.4878 w_k1_MAT_s101
                                 + 1.0879 w_k1_MSFT_s101 + 1.18394 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00259 w_k1_ADM_s101
                                 + 1.29684 w_k1_AXP_s101
                                 + 0.98679 w_k1_DIS_s101
                                 + 0.899809 w_k1_FDX_s101
                                 + 1.58385 w_k1_IBM_s101
                                 + 1.15241 w_k1_INTC_s101
                                 + 1.16129 w_k1_KO_s101 + 1.44046 w_k1_MAT_s101
                                 + 0.944167 w_k1_MSFT_s101
                                 + 1.07658 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.00932 w_k1_ADM_s101
                                 + 0.989341 w_k1_AXP_s101
                                 + 1.26271 w_k1_DIS_s101
                                 + 1.11299 w_k1_FDX_s101
                                 + 0.854244 w_k1_IBM_s101
                                 + 1.09074 w_k1_INTC_s101
                                 + 1.01137 w_k1_KO_s101
                                 + 0.915759 w_k1_MAT_s101
                                 + 1.19646 w_k1_MSFT_s101
                                 + 1.09249 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.815081 w_k1_ADM_s101
                                 + 1.31473 w_k1_AXP_s101
                                 + 1.04961 w_k1_DIS_s101
                                 + 0.97783 w_k1_FDX_s101
                                 + 1.08373 w_k1_IBM_s101
                                 + 0.989887 w_k1_INTC_s101
                                 + 1.06288 w_k1_KO_s101 + 1.51798 w_k1_MAT_s101
                                 + 0.988235 w_k1_MSFT_s101
                                 + 1.06653 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.910844 w_k1_ADM_s101
                                 + 1.08978 w_k1_AXP_s101
                                 + 0.877149 w_k1_DIS_s101
                                 + 0.944651 w_k1_FDX_s101
                                 + 0.690803 w_k1_IBM_s101
                                 + 1.00588 w_k1_INTC_s101
                                 + 0.895695 w_k1_KO_s101
                                 + 1.11185 w_k1_MAT_s101
                                 + 1.01245 w_k1_MSFT_s101
                                 + 0.905649 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.856095 w_k1_ADM_s102
                                 + 0.898312 w_k1_AXP_s102
                                 + 0.84892 w_k1_DIS_s102
                                 + 0.959816 w_k1_FDX_s102
                                 + 1.22439 w_k1_IBM_s102
                                 + 0.793954 w_k1_INTC_s102
                                 + 0.876048 w_k1_KO_s102
                                 + 1.16036 w_k1_MAT_s102
                                 + 0.896818 w_k1_MSFT_s102
                                 + 0.927849 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.759294 w_k1_ADM_s102
                                 + 0.827146 w_k1_AXP_s102
                                 + 0.673971 w_k1_DIS_s102
                                 + 0.808654 w_k1_FDX_s102
                                 + 0.819102 w_k1_IBM_s102
                                 + 0.689706 w_k1_INTC_s102
                                 + 0.663991 w_k1_KO_s102
                                 + 0.955035 w_k1_MAT_s102
                                 + 0.78301 w_k1_MSFT_s102
                                 + 0.800755 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.18021 w_k1_ADM_s102
                                 + 0.885407 w_k1_AXP_s102
                                 + 1.19948 w_k1_DIS_s102 + 1.1834 w_k1_FDX_s102
                                 + 0.728954 w_k1_IBM_s102
                                 + 1.02696 w_k1_INTC_s102
                                 + 0.99049 w_k1_KO_s102
                                 + 0.630835 w_k1_MAT_s102
                                 + 1.12105 w_k1_MSFT_s102
                                 + 0.90395 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.07105 w_k1_ADM_s102
                                 + 1.05898 w_k1_AXP_s102
                                 + 1.29325 w_k1_DIS_s102
                                 + 1.23448 w_k1_FDX_s102
                                 + 1.21093 w_k1_IBM_s102
                                 + 1.10117 w_k1_INTC_s102
                                 + 1.21344 w_k1_KO_s102 + 1.12054 w_k1_MAT_s102
                                 + 1.19149 w_k1_MSFT_s102
                                 + 1.03063 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.44065 w_k1_ADM_s102
                                 + 1.32764 w_k1_AXP_s102
                                 + 1.13075 w_k1_DIS_s102
                                 + 1.13944 w_k1_FDX_s102
                                 + 1.08949 w_k1_IBM_s102
                                 + 0.855054 w_k1_INTC_s102
                                 + 1.02019 w_k1_KO_s102 + 1.29336 w_k1_MAT_s102
                                 + 0.91879 w_k1_MSFT_s102
                                 + 1.00151 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.32722 w_k1_ADM_s102
                                 + 1.35154 w_k1_AXP_s102
                                 + 1.68947 w_k1_DIS_s102
                                 + 1.42228 w_k1_FDX_s102
                                 + 1.52955 w_k1_IBM_s102
                                 + 1.0269 w_k1_INTC_s102 + 1.27771 w_k1_KO_s102
                                 + 1.35458 w_k1_MAT_s102
                                 + 1.16044 w_k1_MSFT_s102
                                 + 1.17653 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00309 w_k1_ADM_s102
                                 + 0.942798 w_k1_AXP_s102
                                 + 0.89234 w_k1_DIS_s102
                                 + 1.07837 w_k1_FDX_s102
                                 + 0.629328 w_k1_IBM_s102
                                 + 0.815779 w_k1_INTC_s102
                                 + 0.70097 w_k1_KO_s102
                                 + 0.713812 w_k1_MAT_s102
                                 + 0.973606 w_k1_MSFT_s102
                                 + 0.767215 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.09357 w_k1_ADM_s102
                                 + 1.08539 w_k1_AXP_s102
                                 + 0.969506 w_k1_DIS_s102
                                 + 0.929898 w_k1_FDX_s102
                                 + 1.56104 w_k1_IBM_s102
                                 + 0.74666 w_k1_INTC_s102
                                 + 1.00701 w_k1_KO_s102 + 1.47744 w_k1_MAT_s102
                                 + 0.869654 w_k1_MSFT_s102
                                 + 0.884289 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.24887 w_k1_ADM_s103
                                 + 1.44575 w_k1_AXP_s103
                                 + 1.23401 w_k1_DIS_s103
                                 + 1.21231 w_k1_FDX_s103
                                 + 1.66325 w_k1_IBM_s103
                                 + 1.10615 w_k1_INTC_s103 + 1.228 w_k1_KO_s103
                                 + 1.72941 w_k1_MAT_s103
                                 + 1.15943 w_k1_MSFT_s103
                                 + 1.30987 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.782547 w_k1_ADM_s103
                                 + 0.838675 w_k1_AXP_s103
                                 + 0.601751 w_k1_DIS_s103
                                 + 0.76232 w_k1_FDX_s103
                                 + 0.965962 w_k1_IBM_s103
                                 + 0.930773 w_k1_INTC_s103
                                 + 0.841604 w_k1_KO_s103
                                 + 0.74676 w_k1_MAT_s103
                                 + 0.79771 w_k1_MSFT_s103
                                 + 0.887176 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.15723 w_k1_ADM_s103
                                 + 1.70702 w_k1_AXP_s103
                                 + 1.91149 w_k1_DIS_s103
                                 + 1.46373 w_k1_FDX_s103
                                 + 1.34201 w_k1_IBM_s103
                                 + 1.33969 w_k1_INTC_s103
                                 + 1.62371 w_k1_KO_s103 + 1.2139 w_k1_MAT_s103
                                 + 1.38238 w_k1_MSFT_s103
                                 + 1.29958 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.840667 w_k1_ADM_s103
                                 + 0.944239 w_k1_AXP_s103
                                 + 0.939684 w_k1_DIS_s103
                                 + 1.01164 w_k1_FDX_s103
                                 + 0.750886 w_k1_IBM_s103
                                 + 0.975072 w_k1_INTC_s103
                                 + 0.95867 w_k1_KO_s103
                                 + 0.849349 w_k1_MAT_s103
                                 + 1.08763 w_k1_MSFT_s103
                                 + 1.00465 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.744662 w_k1_ADM_s103
                                 + 0.598055 w_k1_AXP_s103
                                 + 0.499939 w_k1_DIS_s103
                                 + 0.714174 w_k1_FDX_s103
                                 + 0.654168 w_k1_IBM_s103
                                 + 0.838787 w_k1_INTC_s103
                                 + 0.748549 w_k1_KO_s103
                                 + 0.521623 w_k1_MAT_s103
                                 + 0.792502 w_k1_MSFT_s103
                                 + 0.77934 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.10262 w_k1_ADM_s103
                                 + 1.36461 w_k1_AXP_s103
                                 + 1.39164 w_k1_DIS_s103
                                 + 1.31601 w_k1_FDX_s103
                                 + 0.897507 w_k1_IBM_s103
                                 + 1.40818 w_k1_INTC_s103
                                 + 1.16409 w_k1_KO_s103 + 1.09739 w_k1_MAT_s103
                                 + 1.27608 w_k1_MSFT_s103
                                 + 1.18542 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.00411 w_k1_ADM_s103
                                 + 1.05312 w_k1_AXP_s103
                                 + 0.882459 w_k1_DIS_s103
                                 + 1.01291 w_k1_FDX_s103 + 1.0146 w_k1_IBM_s103
                                 + 1.15382 w_k1_INTC_s103
                                 + 1.10011 w_k1_KO_s103
                                 + 0.884356 w_k1_MAT_s103
                                 + 1.00688 w_k1_MSFT_s103
                                 + 1.02796 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01651 w_k1_ADM_s103
                                 + 1.20608 w_k1_AXP_s103
                                 + 1.03176 w_k1_DIS_s103
                                 + 0.976147 w_k1_FDX_s103
                                 + 1.34274 w_k1_IBM_s103
                                 + 1.14875 w_k1_INTC_s103
                                 + 1.29377 w_k1_KO_s103 + 1.30311 w_k1_MAT_s103
                                 + 1.12104 w_k1_MSFT_s103
                                 + 1.12903 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.21446 w_k1_ADM_s104
                                 + 1.601 w_k1_AXP_s104 + 1.39937 w_k1_DIS_s104
                                 + 1.07949 w_k1_FDX_s104
                                 + 1.46376 w_k1_IBM_s104
                                 + 1.15474 w_k1_INTC_s104
                                 + 1.43215 w_k1_KO_s104 + 1.49909 w_k1_MAT_s104
                                 + 1.05362 w_k1_MSFT_s104
                                 + 1.12339 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.957012 w_k1_ADM_s104
                                 + 1.27511 w_k1_AXP_s104
                                 + 1.27546 w_k1_DIS_s104
                                 + 1.29546 w_k1_FDX_s104
                                 + 1.03187 w_k1_IBM_s104
                                 + 1.19812 w_k1_INTC_s104
                                 + 1.20919 w_k1_KO_s104 + 1.25712 w_k1_MAT_s104
                                 + 1.08561 w_k1_MSFT_s104
                                 + 1.06168 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.961656 w_k1_ADM_s104
                                 + 0.889789 w_k1_AXP_s104
                                 + 0.88927 w_k1_DIS_s104
                                 + 0.878954 w_k1_FDX_s104
                                 + 0.624251 w_k1_IBM_s104
                                 + 0.908064 w_k1_INTC_s104
                                 + 0.815834 w_k1_KO_s104
                                 + 0.620593 w_k1_MAT_s104
                                 + 0.923533 w_k1_MSFT_s104
                                 + 0.832573 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.30774 w_k1_ADM_s104
                                 + 1.296 w_k1_AXP_s104 + 1.07917 w_k1_DIS_s104
                                 + 1.08887 w_k1_FDX_s104
                                 + 0.870382 w_k1_IBM_s104
                                 + 1.11754 w_k1_INTC_s104
                                 + 0.987904 w_k1_KO_s104
                                 + 0.854407 w_k1_MAT_s104
                                 + 0.940768 w_k1_MSFT_s104
                                 + 1.0042 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.848614 w_k1_ADM_s104
                                 + 0.796351 w_k1_AXP_s104
                                 + 0.631236 w_k1_DIS_s104
                                 + 0.759218 w_k1_FDX_s104
                                 + 0.739618 w_k1_IBM_s104
                                 + 0.609051 w_k1_INTC_s104
                                 + 0.668099 w_k1_KO_s104
                                 + 0.750289 w_k1_MAT_s104
                                 + 0.738031 w_k1_MSFT_s104
                                 + 0.814887 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.795564 w_k1_ADM_s104
                                 + 0.801172 w_k1_AXP_s104
                                 + 0.619035 w_k1_DIS_s104
                                 + 0.77262 w_k1_FDX_s104
                                 + 0.934676 w_k1_IBM_s104
                                 + 0.871178 w_k1_INTC_s104
                                 + 0.759424 w_k1_KO_s104
                                 + 1.09072 w_k1_MAT_s104
                                 + 0.700541 w_k1_MSFT_s104
                                 + 0.738343 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.07109 w_k1_ADM_s104
                                 + 1.07927 w_k1_AXP_s104
                                 + 0.726397 w_k1_DIS_s104
                                 + 0.871987 w_k1_FDX_s104
                                 + 0.906159 w_k1_IBM_s104
                                 + 0.811516 w_k1_INTC_s104
                                 + 0.849286 w_k1_KO_s104
                                 + 1.23076 w_k1_MAT_s104
                                 + 0.868584 w_k1_MSFT_s104
                                 + 0.971054 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.813271 w_k1_ADM_s104
                                 + 0.915066 w_k1_AXP_s104
                                 + 1.03219 w_k1_DIS_s104
                                 + 0.94698 w_k1_FDX_s104 + 1.1684 w_k1_IBM_s104
                                 + 1.01248 w_k1_INTC_s104
                                 + 1.16621 w_k1_KO_s104 + 1.18809 w_k1_MAT_s104
                                 + 0.879481 w_k1_MSFT_s104
                                 + 0.904644 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.02666 w_k1_ADM_s105
                                 + 0.890934 w_k1_AXP_s105
                                 + 0.719919 w_k1_DIS_s105
                                 + 0.909267 w_k1_FDX_s105
                                 + 0.661355 w_k1_IBM_s105
                                 + 0.621684 w_k1_INTC_s105
                                 + 0.695465 w_k1_KO_s105
                                 + 0.827488 w_k1_MAT_s105
                                 + 0.76502 w_k1_MSFT_s105
                                 + 0.939595 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.01206 w_k1_ADM_s105
                                 + 0.855458 w_k1_AXP_s105
                                 + 1.03615 w_k1_DIS_s105
                                 + 0.913175 w_k1_FDX_s105
                                 + 1.2458 w_k1_IBM_s105
                                 + 0.78894 w_k1_INTC_s105 + 1.1036 w_k1_KO_s105
                                 + 1.38624 w_k1_MAT_s105
                                 + 0.939338 w_k1_MSFT_s105
                                 + 1.2052 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.09474 w_k1_ADM_s105
                                 + 0.806863 w_k1_AXP_s105
                                 + 0.977153 w_k1_DIS_s105
                                 + 1.05713 w_k1_FDX_s105
                                 + 0.71506 w_k1_IBM_s105
                                 + 0.857725 w_k1_INTC_s105
                                 + 0.718059 w_k1_KO_s105
                                 + 0.631595 w_k1_MAT_s105
                                 + 1.09223 w_k1_MSFT_s105
                                 + 1.00114 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.80085 w_k1_ADM_s105
                                 + 0.715308 w_k1_AXP_s105
                                 + 0.721194 w_k1_DIS_s105
                                 + 0.776055 w_k1_FDX_s105
                                 + 0.891908 w_k1_IBM_s105
                                 + 0.684221 w_k1_INTC_s105
                                 + 0.820153 w_k1_KO_s105
                                 + 0.992773 w_k1_MAT_s105
                                 + 0.763192 w_k1_MSFT_s105
                                 + 1.00434 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.18424 w_k1_ADM_s105
                                 + 1.43237 w_k1_AXP_s105
                                 + 1.54362 w_k1_DIS_s105
                                 + 1.46821 w_k1_FDX_s105
                                 + 1.15772 w_k1_IBM_s105
                                 + 1.01565 w_k1_INTC_s105
                                 + 1.10633 w_k1_KO_s105 + 1.47865 w_k1_MAT_s105
                                 + 1.2423 w_k1_MSFT_s105 + 1.30381 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.931661 w_k1_ADM_s105
                                 + 0.956936 w_k1_AXP_s105
                                 + 0.941998 w_k1_DIS_s105
                                 + 0.978652 w_k1_FDX_s105
                                 + 1.04176 w_k1_IBM_s105
                                 + 0.919375 w_k1_INTC_s105
                                 + 0.913192 w_k1_KO_s105
                                 + 1.06814 w_k1_MAT_s105
                                 + 0.964496 w_k1_MSFT_s105
                                 + 1.06572 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.11355 w_k1_ADM_s105
                                 + 1.13116 w_k1_AXP_s105
                                 + 1.29925 w_k1_DIS_s105
                                 + 1.21192 w_k1_FDX_s105
                                 + 0.990198 w_k1_IBM_s105
                                 + 0.971946 w_k1_INTC_s105
                                 + 1.0816 w_k1_KO_s105 + 1.05096 w_k1_MAT_s105
                                 + 1.16554 w_k1_MSFT_s105
                                 + 1.07984 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.11663 w_k1_ADM_s105
                                 + 1.192 w_k1_AXP_s105 + 1.42956 w_k1_DIS_s105
                                 + 1.07661 w_k1_FDX_s105
                                 + 1.44418 w_k1_IBM_s105
                                 + 0.928531 w_k1_INTC_s105
                                 + 1.30167 w_k1_KO_s105 + 1.81 w_k1_MAT_s105
                                 + 1.04966 w_k1_MSFT_s105 + 1.2552 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.27337 w_k1_ADM_s106
                                 + 0.894194 w_k1_AXP_s106
                                 + 1.35778 w_k1_DIS_s106
                                 + 1.19299 w_k1_FDX_s106
                                 + 0.814829 w_k1_IBM_s106
                                 + 1.28354 w_k1_INTC_s106
                                 + 0.990716 w_k1_KO_s106
                                 + 0.609123 w_k1_MAT_s106
                                 + 1.42166 w_k1_MSFT_s106
                                 + 1.01854 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.860158 w_k1_ADM_s106
                                 + 0.747813 w_k1_AXP_s106
                                 + 0.95804 w_k1_DIS_s106
                                 + 0.919774 w_k1_FDX_s106
                                 + 0.897428 w_k1_IBM_s106
                                 + 0.908911 w_k1_INTC_s106
                                 + 0.987425 w_k1_KO_s106
                                 + 0.921927 w_k1_MAT_s106
                                 + 1.04483 w_k1_MSFT_s106 + 1.0259 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.978944 w_k1_ADM_s106
                                 + 1.27627 w_k1_AXP_s106
                                 + 1.06316 w_k1_DIS_s106
                                 + 0.942667 w_k1_FDX_s106
                                 + 1.47834 w_k1_IBM_s106
                                 + 1.06398 w_k1_INTC_s106
                                 + 1.22627 w_k1_KO_s106 + 1.49792 w_k1_MAT_s106
                                 + 1.17627 w_k1_MSFT_s106
                                 + 1.14394 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.72051 w_k1_ADM_s106
                                 + 1.57486 w_k1_AXP_s106
                                 + 1.26368 w_k1_DIS_s106
                                 + 1.37356 w_k1_FDX_s106
                                 + 1.17671 w_k1_IBM_s106
                                 + 1.04486 w_k1_INTC_s106
                                 + 1.11939 w_k1_KO_s106 + 1.35412 w_k1_MAT_s106
                                 + 1.15896 w_k1_MSFT_s106 + 1.2013 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.06746 w_k1_ADM_s106
                                 + 1.22161 w_k1_AXP_s106
                                 + 1.24697 w_k1_DIS_s106
                                 + 1.15028 w_k1_FDX_s106
                                 + 1.14769 w_k1_IBM_s106
                                 + 0.947476 w_k1_INTC_s106
                                 + 1.07063 w_k1_KO_s106 + 1.1447 w_k1_MAT_s106
                                 + 1.25565 w_k1_MSFT_s106 + 1.0783 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.746893 w_k1_ADM_s106
                                 + 0.753689 w_k1_AXP_s106
                                 + 0.655855 w_k1_DIS_s106
                                 + 0.752732 w_k1_FDX_s106
                                 + 0.647559 w_k1_IBM_s106
                                 + 0.888194 w_k1_INTC_s106
                                 + 0.713598 w_k1_KO_s106
                                 + 0.575351 w_k1_MAT_s106
                                 + 0.803318 w_k1_MSFT_s106
                                 + 0.83493 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.815684 w_k1_ADM_s106
                                 + 0.950044 w_k1_AXP_s106
                                 + 0.706933 w_k1_DIS_s106
                                 + 0.935651 w_k1_FDX_s106
                                 + 0.789864 w_k1_IBM_s106
                                 + 0.987615 w_k1_INTC_s106
                                 + 0.74146 w_k1_KO_s106
                                 + 0.799946 w_k1_MAT_s106
                                 + 0.95561 w_k1_MSFT_s106
                                 + 0.93251 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.936556 w_k1_ADM_s106
                                 + 0.99449 w_k1_AXP_s106
                                 + 0.872488 w_k1_DIS_s106
                                 + 0.954094 w_k1_FDX_s106
                                 + 1.02806 w_k1_IBM_s106
                                 + 0.923272 w_k1_INTC_s106
                                 + 0.86665 w_k1_KO_s106 + 1.00557 w_k1_MAT_s106
                                 + 1.01976 w_k1_MSFT_s106
                                 + 0.91711 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.885858 w_k1_ADM_s107
                                 + 1.29936 w_k1_AXP_s107
                                 + 1.88208 w_k1_DIS_s107
                                 + 1.38731 w_k1_FDX_s107
                                 + 1.42497 w_k1_IBM_s107
                                 + 1.09762 w_k1_INTC_s107
                                 + 1.37285 w_k1_KO_s107 + 1.32721 w_k1_MAT_s107
                                 + 1.19781 w_k1_MSFT_s107
                                 + 1.08481 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.700041 w_k1_ADM_s107
                                 + 0.851476 w_k1_AXP_s107
                                 + 0.893531 w_k1_DIS_s107
                                 + 0.857151 w_k1_FDX_s107
                                 + 1.25917 w_k1_IBM_s107
                                 + 0.934376 w_k1_INTC_s107
                                 + 0.949434 w_k1_KO_s107
                                 + 0.963541 w_k1_MAT_s107
                                 + 0.823587 w_k1_MSFT_s107
                                 + 0.926693 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.03293 w_k1_ADM_s107
                                 + 0.87981 w_k1_AXP_s107
                                 + 1.10169 w_k1_DIS_s107
                                 + 1.02631 w_k1_FDX_s107
                                 + 0.670113 w_k1_IBM_s107
                                 + 0.861577 w_k1_INTC_s107
                                 + 0.812166 w_k1_KO_s107
                                 + 0.62446 w_k1_MAT_s107
                                 + 0.850763 w_k1_MSFT_s107
                                 + 0.901196 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.871138 w_k1_ADM_s107
                                 + 0.657232 w_k1_AXP_s107
                                 + 0.706869 w_k1_DIS_s107
                                 + 0.810029 w_k1_FDX_s107
                                 + 0.737185 w_k1_IBM_s107
                                 + 0.728795 w_k1_INTC_s107
                                 + 0.665131 w_k1_KO_s107
                                 + 0.446011 w_k1_MAT_s107
                                 + 0.737663 w_k1_MSFT_s107
                                 + 0.730809 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1614 w_k1_ADM_s107
                                 + 1.32966 w_k1_AXP_s107
                                 + 1.35132 w_k1_DIS_s107
                                 + 1.40234 w_k1_FDX_s107
                                 + 0.828707 w_k1_IBM_s107
                                 + 1.28991 w_k1_INTC_s107
                                 + 0.997329 w_k1_KO_s107
                                 + 0.807213 w_k1_MAT_s107
                                 + 1.15788 w_k1_MSFT_s107
                                 + 0.862429 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.978783 w_k1_ADM_s107
                                 + 0.965947 w_k1_AXP_s107
                                 + 0.965679 w_k1_DIS_s107
                                 + 0.859971 w_k1_FDX_s107
                                 + 1.00288 w_k1_IBM_s107
                                 + 0.882913 w_k1_INTC_s107
                                 + 0.927549 w_k1_KO_s107
                                 + 0.90158 w_k1_MAT_s107
                                 + 0.864677 w_k1_MSFT_s107
                                 + 0.884344 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.20477 w_k1_ADM_s107
                                 + 1.20767 w_k1_AXP_s107
                                 + 1.30438 w_k1_DIS_s107 + 1.2041 w_k1_FDX_s107
                                 + 0.934485 w_k1_IBM_s107
                                 + 1.16835 w_k1_INTC_s107
                                 + 0.926839 w_k1_KO_s107
                                 + 0.739784 w_k1_MAT_s107
                                 + 1.07325 w_k1_MSFT_s107
                                 + 0.906485 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.886736 w_k1_ADM_s107
                                 + 0.570056 w_k1_AXP_s107
                                 + 1.02083 w_k1_DIS_s107
                                 + 0.874429 w_k1_FDX_s107
                                 + 0.886737 w_k1_IBM_s107
                                 + 0.892635 w_k1_INTC_s107
                                 + 0.877042 w_k1_KO_s107
                                 + 0.709858 w_k1_MAT_s107
                                 + 0.831528 w_k1_MSFT_s107
                                 + 0.805305 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.894717 w_k1_ADM_s110
                                 + 1.15399 w_k1_AXP_s110
                                 + 1.23927 w_k1_DIS_s110
                                 + 1.01941 w_k1_FDX_s110
                                 + 1.57662 w_k1_IBM_s110
                                 + 1.18353 w_k1_INTC_s110
                                 + 1.29973 w_k1_KO_s110 + 1.72858 w_k1_MAT_s110
                                 + 1.18888 w_k1_MSFT_s110
                                 + 0.981431 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.831557 w_k1_ADM_s110
                                 + 0.692763 w_k1_AXP_s110
                                 + 0.643832 w_k1_DIS_s110
                                 + 0.754723 w_k1_FDX_s110
                                 + 0.633649 w_k1_IBM_s110
                                 + 0.870308 w_k1_INTC_s110
                                 + 0.697693 w_k1_KO_s110
                                 + 0.582828 w_k1_MAT_s110
                                 + 0.740155 w_k1_MSFT_s110
                                 + 0.742596 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.878427 w_k1_ADM_s110
                                 + 0.771541 w_k1_AXP_s110
                                 + 0.93868 w_k1_DIS_s110
                                 + 0.900056 w_k1_FDX_s110
                                 + 0.987514 w_k1_IBM_s110
                                 + 0.994935 w_k1_INTC_s110
                                 + 0.969405 w_k1_KO_s110
                                 + 0.838206 w_k1_MAT_s110
                                 + 0.97114 w_k1_MSFT_s110
                                 + 0.898149 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.76883 w_k1_ADM_s110
                                 + 0.92953 w_k1_AXP_s110
                                 + 0.739027 w_k1_DIS_s110
                                 + 0.849163 w_k1_FDX_s110
                                 + 0.8352 w_k1_IBM_s110
                                 + 0.962723 w_k1_INTC_s110
                                 + 0.822038 w_k1_KO_s110
                                 + 0.730985 w_k1_MAT_s110
                                 + 0.833031 w_k1_MSFT_s110
                                 + 0.863608 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.943122 w_k1_ADM_s110
                                 + 1.05332 w_k1_AXP_s110
                                 + 0.939146 w_k1_DIS_s110
                                 + 1.05064 w_k1_FDX_s110
                                 + 0.608993 w_k1_IBM_s110
                                 + 0.998078 w_k1_INTC_s110
                                 + 0.825506 w_k1_KO_s110
                                 + 0.595263 w_k1_MAT_s110
                                 + 1.02459 w_k1_MSFT_s110
                                 + 0.997681 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.34801 w_k1_ADM_s110
                                 + 0.994428 w_k1_AXP_s110
                                 + 1.31746 w_k1_DIS_s110
                                 + 1.26734 w_k1_FDX_s110
                                 + 0.775668 w_k1_IBM_s110
                                 + 1.17307 w_k1_INTC_s110
                                 + 1.15594 w_k1_KO_s110
                                 + 0.701569 w_k1_MAT_s110
                                 + 1.30152 w_k1_MSFT_s110
                                 + 1.05136 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.28795 w_k1_ADM_s110
                                 + 1.34111 w_k1_AXP_s110
                                 + 1.54282 w_k1_DIS_s110
                                 + 1.37871 w_k1_FDX_s110
                                 + 1.07583 w_k1_IBM_s110
                                 + 1.05041 w_k1_INTC_s110
                                 + 1.17301 w_k1_KO_s110 + 1.33892 w_k1_MAT_s110
                                 + 1.24699 w_k1_MSFT_s110
                                 + 1.13215 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.02405 w_k1_ADM_s110
                                 + 1.18161 w_k1_AXP_s110
                                 + 0.944619 w_k1_DIS_s110
                                 + 0.974274 w_k1_FDX_s110
                                 + 1.13529 w_k1_IBM_s110
                                 + 0.895229 w_k1_INTC_s110
                                 + 1.0547 w_k1_KO_s110 + 1.06166 w_k1_MAT_s110
                                 + 0.935015 w_k1_MSFT_s110
                                 + 1.03148 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.86505 w_k1_ADM_s111
                                 + 0.952559 w_k1_AXP_s111
                                 + 0.874417 w_k1_DIS_s111
                                 + 0.940878 w_k1_FDX_s111
                                 + 0.62404 w_k1_IBM_s111
                                 + 0.87451 w_k1_INTC_s111
                                 + 0.770045 w_k1_KO_s111
                                 + 0.767305 w_k1_MAT_s111
                                 + 1.01108 w_k1_MSFT_s111
                                 + 0.937313 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.18778 w_k1_ADM_s111
                                 + 1.24114 w_k1_AXP_s111
                                 + 1.19122 w_k1_DIS_s111
                                 + 1.22462 w_k1_FDX_s111 + 1.1729 w_k1_IBM_s111
                                 + 1.27645 w_k1_INTC_s111
                                 + 1.16817 w_k1_KO_s111 + 1.43122 w_k1_MAT_s111
                                 + 1.31534 w_k1_MSFT_s111
                                 + 1.24067 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.978703 w_k1_ADM_s111
                                 + 1.31284 w_k1_AXP_s111
                                 + 1.36872 w_k1_DIS_s111
                                 + 1.13327 w_k1_FDX_s111
                                 + 1.73527 w_k1_IBM_s111
                                 + 1.29506 w_k1_INTC_s111
                                 + 1.63616 w_k1_KO_s111 + 1.73036 w_k1_MAT_s111
                                 + 1.25879 w_k1_MSFT_s111
                                 + 1.31412 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.906523 w_k1_ADM_s111
                                 + 0.930777 w_k1_AXP_s111
                                 + 0.925678 w_k1_DIS_s111
                                 + 0.834467 w_k1_FDX_s111
                                 + 1.33164 w_k1_IBM_s111
                                 + 0.922602 w_k1_INTC_s111
                                 + 1.0085 w_k1_KO_s111 + 1.54569 w_k1_MAT_s111
                                 + 0.965599 w_k1_MSFT_s111
                                 + 1.03167 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04884 w_k1_ADM_s111
                                 + 0.922744 w_k1_AXP_s111
                                 + 1.13475 w_k1_DIS_s111
                                 + 1.00879 w_k1_FDX_s111
                                 + 0.838557 w_k1_IBM_s111
                                 + 1.01293 w_k1_INTC_s111
                                 + 0.995734 w_k1_KO_s111
                                 + 0.750645 w_k1_MAT_s111
                                 + 1.13592 w_k1_MSFT_s111
                                 + 1.04124 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.663962 w_k1_ADM_s111
                                 + 0.717306 w_k1_AXP_s111
                                 + 0.546352 w_k1_DIS_s111
                                 + 0.66511 w_k1_FDX_s111
                                 + 0.746397 w_k1_IBM_s111
                                 + 0.69459 w_k1_INTC_s111
                                 + 0.638111 w_k1_KO_s111
                                 + 0.798981 w_k1_MAT_s111
                                 + 0.760557 w_k1_MSFT_s111
                                 + 0.776446 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.827291 w_k1_ADM_s111
                                 + 0.872073 w_k1_AXP_s111
                                 + 0.905502 w_k1_DIS_s111
                                 + 0.768561 w_k1_FDX_s111
                                 + 1.06106 w_k1_IBM_s111
                                 + 0.942763 w_k1_INTC_s111
                                 + 1.04118 w_k1_KO_s111 + 0.90718 w_k1_MAT_s111
                                 + 0.952972 w_k1_MSFT_s111
                                 + 0.97171 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.885298 w_k1_ADM_s111
                                 + 1.03438 w_k1_AXP_s111
                                 + 0.727264 w_k1_DIS_s111
                                 + 0.886284 w_k1_FDX_s111
                                 + 0.923692 w_k1_IBM_s111
                                 + 0.858007 w_k1_INTC_s111
                                 + 0.828166 w_k1_KO_s111
                                 + 0.986898 w_k1_MAT_s111
                                 + 0.773335 w_k1_MSFT_s111
                                 + 0.859162 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.764443 w_k1_ADM_s112
                                 + 0.773799 w_k1_AXP_s112
                                 + 0.656251 w_k1_DIS_s112
                                 + 0.698889 w_k1_FDX_s112
                                 + 0.88151 w_k1_IBM_s112
                                 + 0.867269 w_k1_INTC_s112
                                 + 0.778194 w_k1_KO_s112
                                 + 1.10474 w_k1_MAT_s112
                                 + 0.723817 w_k1_MSFT_s112
                                 + 0.90296 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.05473 w_k1_ADM_s112
                                 + 1.49344 w_k1_AXP_s112
                                 + 1.40452 w_k1_DIS_s112
                                 + 1.32039 w_k1_FDX_s112 + 1.0388 w_k1_IBM_s112
                                 + 1.32047 w_k1_INTC_s112
                                 + 1.32926 w_k1_KO_s112 + 1.52272 w_k1_MAT_s112
                                 + 1.17988 w_k1_MSFT_s112
                                 + 1.21472 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.03764 w_k1_ADM_s112
                                 + 1.23655 w_k1_AXP_s112
                                 + 1.44748 w_k1_DIS_s112
                                 + 1.09368 w_k1_FDX_s112
                                 + 1.51126 w_k1_IBM_s112
                                 + 1.17741 w_k1_INTC_s112
                                 + 1.50774 w_k1_KO_s112 + 1.85264 w_k1_MAT_s112
                                 + 1.23245 w_k1_MSFT_s112
                                 + 1.22749 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.918599 w_k1_ADM_s112
                                 + 1.00445 w_k1_AXP_s112
                                 + 1.00254 w_k1_DIS_s112
                                 + 0.910087 w_k1_FDX_s112
                                 + 1.04023 w_k1_IBM_s112
                                 + 1.10297 w_k1_INTC_s112
                                 + 1.18166 w_k1_KO_s112 + 1.37285 w_k1_MAT_s112
                                 + 0.92858 w_k1_MSFT_s112
                                 + 1.08187 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.942374 w_k1_ADM_s112
                                 + 1.0917 w_k1_AXP_s112 + 1.01145 w_k1_DIS_s112
                                 + 0.939777 w_k1_FDX_s112
                                 + 1.33563 w_k1_IBM_s112
                                 + 1.16271 w_k1_INTC_s112 + 1.246 w_k1_KO_s112
                                 + 1.10936 w_k1_MAT_s112
                                 + 0.970842 w_k1_MSFT_s112
                                 + 1.13678 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.794964 w_k1_ADM_s112
                                 + 0.89966 w_k1_AXP_s112
                                 + 0.765264 w_k1_DIS_s112
                                 + 0.84827 w_k1_FDX_s112
                                 + 0.71277 w_k1_IBM_s112
                                 + 0.906656 w_k1_INTC_s112
                                 + 0.813798 w_k1_KO_s112
                                 + 1.08433 w_k1_MAT_s112
                                 + 0.798674 w_k1_MSFT_s112
                                 + 0.87339 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.920024 w_k1_ADM_s112
                                 + 1.37404 w_k1_AXP_s112
                                 + 1.11064 w_k1_DIS_s112
                                 + 1.08367 w_k1_FDX_s112
                                 + 1.19863 w_k1_IBM_s112
                                 + 1.13171 w_k1_INTC_s112
                                 + 1.19261 w_k1_KO_s112 + 1.89225 w_k1_MAT_s112
                                 + 0.964734 w_k1_MSFT_s112
                                 + 0.99547 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.839638 w_k1_ADM_s112
                                 + 1.00798 w_k1_AXP_s112
                                 + 0.860498 w_k1_DIS_s112
                                 + 0.915736 w_k1_FDX_s112
                                 + 1.03665 w_k1_IBM_s112
                                 + 1.04899 w_k1_INTC_s112
                                 + 0.925445 w_k1_KO_s112
                                 + 0.973888 w_k1_MAT_s112
                                 + 0.808835 w_k1_MSFT_s112
                                 + 0.998365 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.15014 w_k1_ADM_s113
                                 + 1.11289 w_k1_AXP_s113
                                 + 1.04715 w_k1_DIS_s113
                                 + 1.12655 w_k1_FDX_s113
                                 + 1.04193 w_k1_IBM_s113
                                 + 0.944081 w_k1_INTC_s113
                                 + 0.952368 w_k1_KO_s113 + 1.0971 w_k1_MAT_s113
                                 + 1.06106 w_k1_MSFT_s113
                                 + 1.11354 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.97966 w_k1_ADM_s113
                                 + 1.07702 w_k1_AXP_s113
                                 + 0.791624 w_k1_DIS_s113
                                 + 0.873421 w_k1_FDX_s113
                                 + 1.15305 w_k1_IBM_s113
                                 + 0.826686 w_k1_INTC_s113
                                 + 0.972518 w_k1_KO_s113
                                 + 1.30438 w_k1_MAT_s113
                                 + 0.919515 w_k1_MSFT_s113
                                 + 1.05567 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.705113 w_k1_ADM_s113
                                 + 0.950721 w_k1_AXP_s113
                                 + 1.06792 w_k1_DIS_s113
                                 + 0.917774 w_k1_FDX_s113
                                 + 1.32495 w_k1_IBM_s113
                                 + 0.850569 w_k1_INTC_s113
                                 + 1.13231 w_k1_KO_s113 + 1.22794 w_k1_MAT_s113
                                 + 0.983839 w_k1_MSFT_s113
                                 + 1.13839 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.816455 w_k1_ADM_s113
                                 + 0.704204 w_k1_AXP_s113
                                 + 0.666207 w_k1_DIS_s113
                                 + 0.747667 w_k1_FDX_s113
                                 + 0.914074 w_k1_IBM_s113
                                 + 0.657824 w_k1_INTC_s113
                                 + 0.809557 w_k1_KO_s113
                                 + 0.720731 w_k1_MAT_s113
                                 + 0.746358 w_k1_MSFT_s113
                                 + 0.908669 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.08308 w_k1_ADM_s113
                                 + 1.1388 w_k1_AXP_s113
                                 + 0.902762 w_k1_DIS_s113
                                 + 1.19825 w_k1_FDX_s113
                                 + 0.687584 w_k1_IBM_s113
                                 + 1.04613 w_k1_INTC_s113
                                 + 0.796927 w_k1_KO_s113
                                 + 0.862565 w_k1_MAT_s113
                                 + 1.00838 w_k1_MSFT_s113
                                 + 1.09666 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.937845 w_k1_ADM_s113
                                 + 0.811336 w_k1_AXP_s113
                                 + 1.16524 w_k1_DIS_s113
                                 + 1.09597 w_k1_FDX_s113
                                 + 0.884204 w_k1_IBM_s113
                                 + 0.787912 w_k1_INTC_s113
                                 + 0.947005 w_k1_KO_s113
                                 + 0.854817 w_k1_MAT_s113
                                 + 1.02889 w_k1_MSFT_s113
                                 + 1.01987 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.01251 w_k1_ADM_s113
                                 + 0.828557 w_k1_AXP_s113
                                 + 0.72139 w_k1_DIS_s113
                                 + 0.868138 w_k1_FDX_s113
                                 + 0.727163 w_k1_IBM_s113
                                 + 0.866028 w_k1_INTC_s113
                                 + 0.830051 w_k1_KO_s113
                                 + 0.709664 w_k1_MAT_s113
                                 + 0.831344 w_k1_MSFT_s113
                                 + 0.991109 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.21308 w_k1_ADM_s113
                                 + 1.46368 w_k1_AXP_s113
                                 + 1.40864 w_k1_DIS_s113
                                 + 1.19673 w_k1_FDX_s113
                                 + 1.35057 w_k1_IBM_s113
                                 + 1.02967 w_k1_INTC_s113
                                 + 1.41923 w_k1_KO_s113 + 1.78325 w_k1_MAT_s113
                                 + 1.16199 w_k1_MSFT_s113
                                 + 1.21275 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.78098 w_k1_ADM_s114
                                 + 0.81928 w_k1_AXP_s114 + 1.0048 w_k1_DIS_s114
                                 + 1.08421 w_k1_FDX_s114
                                 + 0.707547 w_k1_IBM_s114
                                 + 1.02233 w_k1_INTC_s114
                                 + 0.743187 w_k1_KO_s114
                                 + 0.556706 w_k1_MAT_s114
                                 + 0.994657 w_k1_MSFT_s114
                                 + 1.00778 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.737279 w_k1_ADM_s114
                                 + 0.675925 w_k1_AXP_s114
                                 + 0.750036 w_k1_DIS_s114
                                 + 0.770162 w_k1_FDX_s114
                                 + 0.980337 w_k1_IBM_s114
                                 + 0.905383 w_k1_INTC_s114
                                 + 0.894358 w_k1_KO_s114
                                 + 1.02499 w_k1_MAT_s114
                                 + 0.883497 w_k1_MSFT_s114
                                 + 0.943018 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.24742 w_k1_ADM_s114
                                 + 1.21389 w_k1_AXP_s114
                                 + 1.51228 w_k1_DIS_s114
                                 + 1.43279 w_k1_FDX_s114
                                 + 0.966746 w_k1_IBM_s114
                                 + 1.26209 w_k1_INTC_s114
                                 + 1.03328 w_k1_KO_s114 + 1.18867 w_k1_MAT_s114
                                 + 1.3947 w_k1_MSFT_s114 + 1.11504 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.821341 w_k1_ADM_s114
                                 + 0.652786 w_k1_AXP_s114
                                 + 0.650068 w_k1_DIS_s114
                                 + 0.797899 w_k1_FDX_s114
                                 + 0.778548 w_k1_IBM_s114
                                 + 0.909788 w_k1_INTC_s114
                                 + 0.766963 w_k1_KO_s114
                                 + 0.926585 w_k1_MAT_s114
                                 + 0.874467 w_k1_MSFT_s114
                                 + 0.882951 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.01815 w_k1_ADM_s114
                                 + 1.09669 w_k1_AXP_s114 + 1.1439 w_k1_DIS_s114
                                 + 1.12902 w_k1_FDX_s114
                                 + 0.942925 w_k1_IBM_s114
                                 + 1.14376 w_k1_INTC_s114
                                 + 0.900249 w_k1_KO_s114
                                 + 0.943773 w_k1_MAT_s114
                                 + 0.991746 w_k1_MSFT_s114
                                 + 1.05842 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.81302 w_k1_ADM_s114
                                 + 1.15068 w_k1_AXP_s114
                                 + 1.36763 w_k1_DIS_s114
                                 + 1.07229 w_k1_FDX_s114
                                 + 1.31313 w_k1_IBM_s114
                                 + 0.970412 w_k1_INTC_s114
                                 + 1.09141 w_k1_KO_s114 + 1.57554 w_k1_MAT_s114
                                 + 1.11293 w_k1_MSFT_s114
                                 + 1.10824 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.860662 w_k1_ADM_s114
                                 + 0.914093 w_k1_AXP_s114
                                 + 0.95766 w_k1_DIS_s114
                                 + 0.86686 w_k1_FDX_s114
                                 + 1.36758 w_k1_IBM_s114
                                 + 0.943516 w_k1_INTC_s114
                                 + 1.14855 w_k1_KO_s114 + 1.42737 w_k1_MAT_s114
                                 + 0.983548 w_k1_MSFT_s114
                                 + 1.0983 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.900137 w_k1_ADM_s114
                                 + 1.05016 w_k1_AXP_s114
                                 + 0.920406 w_k1_DIS_s114
                                 + 1.01509 w_k1_FDX_s114
                                 + 0.948212 w_k1_IBM_s114
                                 + 0.853033 w_k1_INTC_s114
                                 + 0.929736 w_k1_KO_s114
                                 + 1.59919 w_k1_MAT_s114
                                 + 0.973554 w_k1_MSFT_s114
                                 + 1.06377 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.13092 w_k1_ADM_s115
                                 + 0.91396 w_k1_AXP_s115
                                 + 0.972738 w_k1_DIS_s115
                                 + 0.960924 w_k1_FDX_s115
                                 + 0.985266 w_k1_IBM_s115
                                 + 1.21525 w_k1_INTC_s115
                                 + 1.08711 w_k1_KO_s115 + 1.05577 w_k1_MAT_s115
                                 + 0.928245 w_k1_MSFT_s115
                                 + 1.07276 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.969076 w_k1_ADM_s115
                                 + 0.737304 w_k1_AXP_s115
                                 + 0.815918 w_k1_DIS_s115
                                 + 0.891234 w_k1_FDX_s115
                                 + 0.744076 w_k1_IBM_s115
                                 + 0.96157 w_k1_INTC_s115
                                 + 0.78728 w_k1_KO_s115
                                 + 0.847849 w_k1_MAT_s115
                                 + 0.880638 w_k1_MSFT_s115
                                 + 0.895986 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.37694 w_k1_ADM_s115
                                 + 1.35115 w_k1_AXP_s115
                                 + 1.54455 w_k1_DIS_s115
                                 + 1.43227 w_k1_FDX_s115
                                 + 1.19382 w_k1_IBM_s115
                                 + 1.31247 w_k1_INTC_s115
                                 + 1.26358 w_k1_KO_s115
                                 + 0.972573 w_k1_MAT_s115
                                 + 1.22632 w_k1_MSFT_s115
                                 + 1.30171 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.47384 w_k1_ADM_s115
                                 + 1.70765 w_k1_AXP_s115 + 1.4478 w_k1_DIS_s115
                                 + 1.35202 w_k1_FDX_s115
                                 + 1.63807 w_k1_IBM_s115
                                 + 1.42087 w_k1_INTC_s115
                                 + 1.34731 w_k1_KO_s115 + 2.49519 w_k1_MAT_s115
                                 + 1.27804 w_k1_MSFT_s115
                                 + 1.28182 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.635227 w_k1_ADM_s115
                                 + 0.563298 w_k1_AXP_s115
                                 + 0.556359 w_k1_DIS_s115
                                 + 0.703469 w_k1_FDX_s115
                                 + 0.75733 w_k1_IBM_s115
                                 + 0.780743 w_k1_INTC_s115
                                 + 0.728018 w_k1_KO_s115
                                 + 0.626229 w_k1_MAT_s115
                                 + 0.729519 w_k1_MSFT_s115
                                 + 0.828541 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.38083 w_k1_ADM_s115
                                 + 0.994246 w_k1_AXP_s115
                                 + 0.87745 w_k1_DIS_s115
                                 + 1.18314 w_k1_FDX_s115
                                 + 0.640984 w_k1_IBM_s115
                                 + 1.07696 w_k1_INTC_s115
                                 + 0.846419 w_k1_KO_s115
                                 + 0.530979 w_k1_MAT_s115
                                 + 0.963683 w_k1_MSFT_s115
                                 + 0.967002 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.11767 w_k1_ADM_s115
                                 + 1.0988 w_k1_AXP_s115
                                 + 0.982526 w_k1_DIS_s115
                                 + 0.984457 w_k1_FDX_s115
                                 + 1.40069 w_k1_IBM_s115
                                 + 1.11543 w_k1_INTC_s115 + 1.0883 w_k1_KO_s115
                                 + 1.11892 w_k1_MAT_s115
                                 + 0.854596 w_k1_MSFT_s115
                                 + 1.03949 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.882572 w_k1_ADM_s115
                                 + 0.732209 w_k1_AXP_s115
                                 + 0.781862 w_k1_DIS_s115
                                 + 0.748254 w_k1_FDX_s115
                                 + 1.2004 w_k1_IBM_s115
                                 + 0.87448 w_k1_INTC_s115
                                 + 0.983667 w_k1_KO_s115
                                 + 0.963548 w_k1_MAT_s115
                                 + 0.768711 w_k1_MSFT_s115
                                 + 0.884334 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.51162 w_k1_ADM_s116
                                 + 1.22389 w_k1_AXP_s116 + 1.1005 w_k1_DIS_s116
                                 + 1.07963 w_k1_FDX_s116
                                 + 1.23622 w_k1_IBM_s116
                                 + 1.18625 w_k1_INTC_s116
                                 + 1.00896 w_k1_KO_s116 + 1.15325 w_k1_MAT_s116
                                 + 1.10483 w_k1_MSFT_s116
                                 + 0.945249 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.928927 w_k1_ADM_s116
                                 + 0.614467 w_k1_AXP_s116
                                 + 0.553475 w_k1_DIS_s116
                                 + 0.679945 w_k1_FDX_s116
                                 + 0.802836 w_k1_IBM_s116
                                 + 0.837258 w_k1_INTC_s116
                                 + 0.671658 w_k1_KO_s116
                                 + 0.617325 w_k1_MAT_s116
                                 + 0.779472 w_k1_MSFT_s116
                                 + 0.744775 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.838354 w_k1_ADM_s116
                                 + 0.906652 w_k1_AXP_s116
                                 + 0.701751 w_k1_DIS_s116
                                 + 0.660772 w_k1_FDX_s116
                                 + 1.26544 w_k1_IBM_s116
                                 + 0.917141 w_k1_INTC_s116
                                 + 0.95592 w_k1_KO_s116 + 1.03044 w_k1_MAT_s116
                                 + 0.937191 w_k1_MSFT_s116
                                 + 0.985655 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.45644 w_k1_ADM_s116
                                 + 1.05338 w_k1_AXP_s116
                                 + 1.07622 w_k1_DIS_s116
                                 + 1.04707 w_k1_FDX_s116
                                 + 0.97718 w_k1_IBM_s116
                                 + 0.977259 w_k1_INTC_s116
                                 + 0.993357 w_k1_KO_s116
                                 + 0.956178 w_k1_MAT_s116
                                 + 0.933977 w_k1_MSFT_s116
                                 + 1.09088 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.19278 w_k1_ADM_s116
                                 + 0.945102 w_k1_AXP_s116
                                 + 0.843632 w_k1_DIS_s116
                                 + 1.00063 w_k1_FDX_s116
                                 + 0.731058 w_k1_IBM_s116
                                 + 0.899625 w_k1_INTC_s116
                                 + 0.723595 w_k1_KO_s116
                                 + 0.69157 w_k1_MAT_s116
                                 + 0.91862 w_k1_MSFT_s116
                                 + 0.942788 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.33663 w_k1_ADM_s116
                                 + 1.31949 w_k1_AXP_s116
                                 + 1.28769 w_k1_DIS_s116
                                 + 1.29071 w_k1_FDX_s116
                                 + 0.836866 w_k1_IBM_s116
                                 + 1.25784 w_k1_INTC_s116
                                 + 0.978737 w_k1_KO_s116 + 0.982 w_k1_MAT_s116
                                 + 1.28152 w_k1_MSFT_s116
                                 + 1.18717 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.68034 w_k1_ADM_s116
                                 + 1.69028 w_k1_AXP_s116
                                 + 1.39698 w_k1_DIS_s116
                                 + 1.30138 w_k1_FDX_s116
                                 + 1.73371 w_k1_IBM_s116
                                 + 1.15421 w_k1_INTC_s116
                                 + 1.27896 w_k1_KO_s116 + 1.9038 w_k1_MAT_s116
                                 + 1.09909 w_k1_MSFT_s116
                                 + 1.27437 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.45935 w_k1_ADM_s116
                                 + 1.56897 w_k1_AXP_s116
                                 + 1.60875 w_k1_DIS_s116
                                 + 1.37432 w_k1_FDX_s116
                                 + 1.12557 w_k1_IBM_s116
                                 + 1.32538 w_k1_INTC_s116
                                 + 1.21821 w_k1_KO_s116 + 1.51666 w_k1_MAT_s116
                                 + 1.19585 w_k1_MSFT_s116 + 1.229 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.25091 w_k1_ADM_s117
                                 + 0.971623 w_k1_AXP_s117
                                 + 1.08549 w_k1_DIS_s117 + 1.1076 w_k1_FDX_s117
                                 + 0.872616 w_k1_IBM_s117
                                 + 1.13221 w_k1_INTC_s117 + 1.0442 w_k1_KO_s117
                                 + 0.746564 w_k1_MAT_s117
                                 + 1.09241 w_k1_MSFT_s117
                                 + 1.09974 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.22832 w_k1_ADM_s117
                                 + 1.08973 w_k1_AXP_s117
                                 + 1.02379 w_k1_DIS_s117
                                 + 1.13074 w_k1_FDX_s117
                                 + 0.653444 w_k1_IBM_s117
                                 + 1.16573 w_k1_INTC_s117
                                 + 0.890374 w_k1_KO_s117
                                 + 0.991573 w_k1_MAT_s117
                                 + 1.1265 w_k1_MSFT_s117 + 1.09493 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.980059 w_k1_ADM_s117
                                 + 0.866922 w_k1_AXP_s117
                                 + 0.958539 w_k1_DIS_s117
                                 + 0.891674 w_k1_FDX_s117
                                 + 1.31561 w_k1_IBM_s117
                                 + 1.02762 w_k1_INTC_s117 + 1.212 w_k1_KO_s117
                                 + 1.12165 w_k1_MAT_s117
                                 + 0.980168 w_k1_MSFT_s117
                                 + 0.949124 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.6754 w_k1_ADM_s117
                                 + 1.19403 w_k1_AXP_s117
                                 + 1.15007 w_k1_DIS_s117
                                 + 1.01049 w_k1_FDX_s117
                                 + 1.69069 w_k1_IBM_s117
                                 + 1.13365 w_k1_INTC_s117
                                 + 1.54327 w_k1_KO_s117 + 1.63065 w_k1_MAT_s117
                                 + 1.0737 w_k1_MSFT_s117 + 1.34226 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.16414 w_k1_ADM_s117
                                 + 0.757242 w_k1_AXP_s117
                                 + 0.92098 w_k1_DIS_s117
                                 + 0.93859 w_k1_FDX_s117
                                 + 0.749393 w_k1_IBM_s117
                                 + 0.944172 w_k1_INTC_s117
                                 + 0.999064 w_k1_KO_s117
                                 + 0.649651 w_k1_MAT_s117
                                 + 1.04886 w_k1_MSFT_s117
                                 + 0.953272 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.850054 w_k1_ADM_s117
                                 + 0.683961 w_k1_AXP_s117
                                 + 0.655141 w_k1_DIS_s117
                                 + 0.751923 w_k1_FDX_s117
                                 + 0.806711 w_k1_IBM_s117
                                 + 0.923816 w_k1_INTC_s117
                                 + 0.900651 w_k1_KO_s117
                                 + 0.560203 w_k1_MAT_s117
                                 + 0.883551 w_k1_MSFT_s117
                                 + 0.923624 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.1902 w_k1_ADM_s117
                                 + 1.08555 w_k1_AXP_s117
                                 + 0.992028 w_k1_DIS_s117
                                 + 1.03985 w_k1_FDX_s117 + 1.0587 w_k1_IBM_s117
                                 + 1.25179 w_k1_INTC_s117
                                 + 1.00912 w_k1_KO_s117
                                 + 0.993675 w_k1_MAT_s117
                                 + 1.20482 w_k1_MSFT_s117
                                 + 1.03343 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.35979 w_k1_ADM_s117
                                 + 1.16666 w_k1_AXP_s117
                                 + 1.70735 w_k1_DIS_s117
                                 + 1.24237 w_k1_FDX_s117
                                 + 1.25129 w_k1_IBM_s117
                                 + 1.28693 w_k1_INTC_s117
                                 + 1.49287 w_k1_KO_s117 + 1.18544 w_k1_MAT_s117
                                 + 1.39471 w_k1_MSFT_s117
                                 + 1.15396 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0344442754696835 w_k1_ADM_s000
                                 + 0.0341713461250686 w_k1_ADM_s001
                                 + 0.0450178328332661 w_k1_ADM_s002
                                 + 0.0436235897092463 w_k1_ADM_s003
                                 + 0.0224779683585169 w_k1_ADM_s004
                                 + 0.0450453149045493 w_k1_ADM_s005
                                 + 0.0221113017530255 w_k1_ADM_s006
                                 + 0.029550435728948 w_k1_ADM_s007
                                 + 0.0180892188823785 w_k1_ADM_s010
                                 + 0.0201574809267469 w_k1_ADM_s011
                                 + 0.0378443402607121 w_k1_ADM_s012
                                 + 0.0213746314447249 w_k1_ADM_s013
                                 + 0.044146269943991 w_k1_ADM_s014
                                 + 0.0646889013386149 w_k1_ADM_s015
                                 + 0.0159500303680156 w_k1_ADM_s016
                                 + 0.0472113545524145 w_k1_ADM_s017
                                 + 0.0108722314201391 w_k1_ADM_s100
                                 + 0.00960927438081937 w_k1_ADM_s101
                                 + 0.0394878768007826 w_k1_ADM_s102
                                 + 0.0347147300613045 w_k1_ADM_s103
                                 + 0.0163022986626233 w_k1_ADM_s104
                                 + 0.00856672469684437 w_k1_ADM_s105
                                 + 0.0101471221957402 w_k1_ADM_s106
                                 + 0.007868567289664 w_k1_ADM_s107
                                 + 0.0837509530187837 w_k1_ADM_s110
                                 + 0.0324519438242854 w_k1_ADM_s111
                                 + 0.0224613839870473 w_k1_ADM_s112
                                 + 0.0381089124681869 w_k1_ADM_s113
                                 + 0.0229357443874446 w_k1_ADM_s114
                                 + 0.0422851526952297 w_k1_ADM_s115
                                 + 0.035356358396765 w_k1_ADM_s116
                                 + 0.0583077147848523 w_k1_ADM_s117
                                 + 0.0293124042355155 w_k1_AXP_s000
                                 + 0.0365736247344455 w_k1_AXP_s001
                                 + 0.0434556854908195 w_k1_AXP_s002
                                 + 0.0436239889281208 w_k1_AXP_s003
                                 + 0.0200217714490505 w_k1_AXP_s004
                                 + 0.0455847225847207 w_k1_AXP_s005
                                 + 0.0220797847268364 w_k1_AXP_s006
                                 + 0.0299357912909885 w_k1_AXP_s007
                                 + 0.0222536515524802 w_k1_AXP_s010
                                 + 0.0203596804946496 w_k1_AXP_s011
                                 + 0.0362895449848108 w_k1_AXP_s012
                                 + 0.0225610860822242 w_k1_AXP_s013
                                 + 0.0432843083042219 w_k1_AXP_s014
                                 + 0.0656422619256356 w_k1_AXP_s015
                                 + 0.018081592546565 w_k1_AXP_s016
                                 + 0.0491547404547531 w_k1_AXP_s017
                                 + 0.0132881734659995 w_k1_AXP_s100
                                 + 0.0121098912651388 w_k1_AXP_s101
                                 + 0.0386730323037049 w_k1_AXP_s102
                                 + 0.0389560778714164 w_k1_AXP_s103
                                 + 0.0184684777066419 w_k1_AXP_s104
                                 + 0.00838748232258001 w_k1_AXP_s105
                                 + 0.0106689606747352 w_k1_AXP_s106
                                 + 0.00849253686091447 w_k1_AXP_s107
                                 + 0.088999505047346 w_k1_AXP_s110
                                 + 0.0341846580952225 w_k1_AXP_s111
                                 + 0.0276095440952625 w_k1_AXP_s112
                                 + 0.0386179338156357 w_k1_AXP_s113
                                 + 0.0246270556980042 w_k1_AXP_s114
                                 + 0.0370617729679584 w_k1_AXP_s115
                                 + 0.0306771325349933 w_k1_AXP_s116
                                 + 0.0475498155124452 w_k1_AXP_s117
                                 + 0.028094645571009 w_k1_DIS_s000
                                 + 0.0346982966770214 w_k1_DIS_s001
                                 + 0.0460789750638598 w_k1_DIS_s002
                                 + 0.0475095043362169 w_k1_DIS_s003
                                 + 0.0194102730167773 w_k1_DIS_s004
                                 + 0.0455406209452014 w_k1_DIS_s005
                                 + 0.0217294425882095 w_k1_DIS_s006
                                 + 0.0294031852569149 w_k1_DIS_s007
                                 + 0.0221925150036284 w_k1_DIS_s010
                                 + 0.0177955343350659 w_k1_DIS_s011
                                 + 0.0380983162400363 w_k1_DIS_s012
                                 + 0.0216646203931985 w_k1_DIS_s013
                                 + 0.0427238524552842 w_k1_DIS_s014
                                 + 0.0689447134095833 w_k1_DIS_s015
                                 + 0.0192450762586107 w_k1_DIS_s016
                                 + 0.0486498587771545 w_k1_DIS_s017
                                 + 0.0124780432129335 w_k1_DIS_s100
                                 + 0.011381251606327 w_k1_DIS_s101
                                 + 0.0394359026396336 w_k1_DIS_s102
                                 + 0.0353851322881161 w_k1_DIS_s103
                                 + 0.0170524816124242 w_k1_DIS_s104
                                 + 0.00907868371806412 w_k1_DIS_s105
                                 + 0.00997926012063518 w_k1_DIS_s106
                                 + 0.0104835278931581 w_k1_DIS_s107
                                 + 0.0882398230318236 w_k1_DIS_s110
                                 + 0.0338071674197029 w_k1_DIS_s111
                                 + 0.0257808394917196 w_k1_DIS_s112
                                 + 0.035933731703378 w_k1_DIS_s113
                                 + 0.0270382434427347 w_k1_DIS_s114
                                 + 0.0372894398558359 w_k1_DIS_s115
                                 + 0.028057013259158 w_k1_DIS_s116
                                 + 0.051726104078385 w_k1_DIS_s117
                                 + 0.0284134271903691 w_k1_FDX_s000
                                 + 0.036752434129676 w_k1_FDX_s001
                                 + 0.0435843699614729 w_k1_FDX_s002
                                 + 0.0460091356770987 w_k1_FDX_s003
                                 + 0.0201443656587044 w_k1_FDX_s004
                                 + 0.0451760270106328 w_k1_FDX_s005
                                 + 0.0205647509344652 w_k1_FDX_s006
                                 + 0.0289159602154239 w_k1_FDX_s007
                                 + 0.0202852808237936 w_k1_FDX_s010
                                 + 0.0196727249008134 w_k1_FDX_s011
                                 + 0.0363874931744482 w_k1_FDX_s012
                                 + 0.0215235309707949 w_k1_FDX_s013
                                 + 0.0421291883744193 w_k1_FDX_s014
                                 + 0.0650542885861115 w_k1_FDX_s015
                                 + 0.0175768926797812 w_k1_FDX_s016
                                 + 0.0471057139061132 w_k1_FDX_s017
                                 + 0.0122724193508935 w_k1_FDX_s100
                                 + 0.0103812657207485 w_k1_FDX_s101
                                 + 0.0405365277387736 w_k1_FDX_s102
                                 + 0.0368553915199772 w_k1_FDX_s103
                                 + 0.0164867528317898 w_k1_FDX_s104
                                 + 0.00876193506052109 w_k1_FDX_s105
                                 + 0.0102622210783703 w_k1_FDX_s106
                                 + 0.00907187234775692 w_k1_FDX_s107
                                 + 0.0866619893606873 w_k1_FDX_s110
                                 + 0.0329231304942294 w_k1_FDX_s111
                                 + 0.0244561595555543 w_k1_FDX_s112
                                 + 0.0372441061426886 w_k1_FDX_s113
                                 + 0.0263720162293855 w_k1_FDX_s114
                                 + 0.0387748928396935 w_k1_FDX_s115
                                 + 0.0283580519187472 w_k1_FDX_s116
                                 + 0.0499438581746092 w_k1_FDX_s117
                                 + 0.0291728549080323 w_k1_IBM_s000
                                 + 0.0374157173129142 w_k1_IBM_s001
                                 + 0.0432282377659919 w_k1_IBM_s002
                                 + 0.0463565796688885 w_k1_IBM_s003
                                 + 0.020087081642301 w_k1_IBM_s004
                                 + 0.0453708090372437 w_k1_IBM_s005
                                 + 0.0214443308026279 w_k1_IBM_s006
                                 + 0.0276950479525562 w_k1_IBM_s007
                                 + 0.0213801108541756 w_k1_IBM_s010
                                 + 0.0200789885847216 w_k1_IBM_s011
                                 + 0.0399055749291177 w_k1_IBM_s012
                                 + 0.02261774583922 w_k1_IBM_s013
                                 + 0.0426815249311935 w_k1_IBM_s014
                                 + 0.0652987432498607 w_k1_IBM_s015
                                 + 0.0171630204838616 w_k1_IBM_s016
                                 + 0.0516100679940056 w_k1_IBM_s017
                                 + 0.0125995666359286 w_k1_IBM_s100
                                 + 0.0102583843521368 w_k1_IBM_s101
                                 + 0.039959145328188 w_k1_IBM_s102
                                 + 0.0369258405532293 w_k1_IBM_s103
                                 + 0.017417640871076 w_k1_IBM_s104
                                 + 0.00851000676783199 w_k1_IBM_s105
                                 + 0.010456332103081 w_k1_IBM_s106
                                 + 0.00867517160595873 w_k1_IBM_s107
                                 + 0.0867267918583771 w_k1_IBM_s110
                                 + 0.0346294826354771 w_k1_IBM_s111
                                 + 0.0259144167214364 w_k1_IBM_s112
                                 + 0.0382146464846155 w_k1_IBM_s113
                                 + 0.0265326828419953 w_k1_IBM_s114
                                 + 0.0386466726016472 w_k1_IBM_s115
                                 + 0.0293165668794717 w_k1_IBM_s116
                                 + 0.0504526065813496 w_k1_IBM_s117
                                 + 0.0260298052988733 w_k1_INTC_s000
                                 + 0.0403748027320604 w_k1_INTC_s001
                                 + 0.0442108015696122 w_k1_INTC_s002
                                 + 0.0444580982680066 w_k1_INTC_s003
                                 + 0.0182103356871899 w_k1_INTC_s004
                                 + 0.0451555949130944 w_k1_INTC_s005
                                 + 0.0203433143651494 w_k1_INTC_s006
                                 + 0.0287461740087751 w_k1_INTC_s007
                                 + 0.0238554716859738 w_k1_INTC_s010
                                 + 0.0189051221205514 w_k1_INTC_s011
                                 + 0.0372002046095805 w_k1_INTC_s012
                                 + 0.025109368915502 w_k1_INTC_s013
                                 + 0.0427788491756349 w_k1_INTC_s014
                                 + 0.0561690916831947 w_k1_INTC_s015
                                 + 0.0194221572116878 w_k1_INTC_s016
                                 + 0.0491667125749096 w_k1_INTC_s017
                                 + 0.0123278883686223 w_k1_INTC_s100
                                 + 0.0106506671081022 w_k1_INTC_s101
                                 + 0.0328184501368258 w_k1_INTC_s102
                                 + 0.0395968672265146 w_k1_INTC_s103
                                 + 0.0165391823057914 w_k1_INTC_s104
                                 + 0.00719953211220061 w_k1_INTC_s105
                                 + 0.0100578694846697 w_k1_INTC_s106
                                 + 0.00819530913994034 w_k1_INTC_s107
                                 + 0.0863800454454305 w_k1_INTC_s110
                                 + 0.0343392481337533 w_k1_INTC_s111
                                 + 0.0269919883034818 w_k1_INTC_s112
                                 + 0.0329529657580862 w_k1_INTC_s113
                                 + 0.0258299191699032 w_k1_INTC_s114
                                 + 0.0411901461575799 w_k1_INTC_s115
                                 + 0.0289543351351558 w_k1_INTC_s116
                                 + 0.0550528653341344 w_k1_INTC_s117
                                 + 0.0295768413069049 w_k1_KO_s000
                                 + 0.0364643642179186 w_k1_KO_s001
                                 + 0.0437738582267308 w_k1_KO_s002
                                 + 0.0490992873592666 w_k1_KO_s003
                                 + 0.0224814893209179 w_k1_KO_s004
                                 + 0.0443234908312217 w_k1_KO_s005
                                 + 0.0212644773813709 w_k1_KO_s006
                                 + 0.0326166025185074 w_k1_KO_s007
                                 + 0.0233585451442354 w_k1_KO_s010
                                 + 0.0200516023047594 w_k1_KO_s011
                                 + 0.0373422159666239 w_k1_KO_s012
                                 + 0.020004300555329 w_k1_KO_s013
                                 + 0.0427729437124366 w_k1_KO_s014
                                 + 0.0677052837795362 w_k1_KO_s015
                                 + 0.0172883288988155 w_k1_KO_s016
                                 + 0.0477208529882231 w_k1_KO_s017
                                 + 0.0130679005081739 w_k1_KO_s100
                                 + 0.0109292897137053 w_k1_KO_s101
                                 + 0.0350677315905644 w_k1_KO_s102
                                 + 0.0389959970040351 w_k1_KO_s103
                                 + 0.0177047771911795 w_k1_KO_s104
                                 + 0.00814166408066193 w_k1_KO_s105
                                 + 0.00977858895844557 w_k1_KO_s106
                                 + 0.00836560381281604 w_k1_KO_s107
                                 + 0.0871578695023952 w_k1_KO_s110
                                 + 0.0343052355524631 w_k1_KO_s111
                                 + 0.0275284056015685 w_k1_KO_s112
                                 + 0.0371264588544096 w_k1_KO_s113
                                 + 0.0246128152190846 w_k1_KO_s114
                                 + 0.0377826592298734 w_k1_KO_s115
                                 + 0.0262191289490334 w_k1_KO_s116
                                 + 0.0548292799191269 w_k1_KO_s117
                                 + 0.0294661479759292 w_k1_MAT_s000
                                 + 0.0334106219378801 w_k1_MAT_s001
                                 + 0.0547592143168207 w_k1_MAT_s002
                                 + 0.0635651227353535 w_k1_MAT_s003
                                 + 0.0201522440246007 w_k1_MAT_s004
                                 + 0.0439663482666806 w_k1_MAT_s005
                                 + 0.0251472666437655 w_k1_MAT_s006
                                 + 0.0417831475840521 w_k1_MAT_s007
                                 + 0.0230826626479078 w_k1_MAT_s010
                                 + 0.0216308716261907 w_k1_MAT_s011
                                 + 0.0412873592619709 w_k1_MAT_s012
                                 + 0.0181755973834263 w_k1_MAT_s013
                                 + 0.0517835984304562 w_k1_MAT_s014
                                 + 0.0764168795198488 w_k1_MAT_s015
                                 + 0.0167022183834104 w_k1_MAT_s016
                                 + 0.0451936513160009 w_k1_MAT_s017
                                 + 0.0128462341542334 w_k1_MAT_s100
                                 + 0.0132843815347338 w_k1_MAT_s101
                                 + 0.0403664330991826 w_k1_MAT_s102
                                 + 0.0352756653914489 w_k1_MAT_s103
                                 + 0.018766623617344 w_k1_MAT_s104
                                 + 0.00954915818844699 w_k1_MAT_s105
                                 + 0.0104397096180907 w_k1_MAT_s106
                                 + 0.0074854242174706 w_k1_MAT_s107
                                 + 0.0856183436513213 w_k1_MAT_s110
                                 + 0.0371888165197953 w_k1_MAT_s111
                                 + 0.0329684006199398 w_k1_MAT_s112
                                 + 0.0409313005857709 w_k1_MAT_s113
                                 + 0.0308544887554825 w_k1_MAT_s114
                                 + 0.0362623232178003 w_k1_MAT_s115
                                 + 0.0289587886446051 w_k1_MAT_s116
                                 + 0.04614217830457 w_k1_MAT_s117
                                 + 0.0285966394592258 w_k1_MSFT_s000
                                 + 0.0347331761653355 w_k1_MSFT_s001
                                 + 0.0452696160076091 w_k1_MSFT_s002
                                 + 0.0479056726088187 w_k1_MSFT_s003
                                 + 0.0197147497627393 w_k1_MSFT_s004
                                 + 0.0470905322208547 w_k1_MSFT_s005
                                 + 0.0209525749558392 w_k1_MSFT_s006
                                 + 0.0271089493829665 w_k1_MSFT_s007
                                 + 0.0215248591582268 w_k1_MSFT_s010
                                 + 0.0205846941333416 w_k1_MSFT_s011
                                 + 0.0374290893749566 w_k1_MSFT_s012
                                 + 0.0224351755967602 w_k1_MSFT_s013
                                 + 0.0426272151927192 w_k1_MSFT_s014
                                 + 0.0710315850015385 w_k1_MSFT_s015
                                 + 0.0176811376510718 w_k1_MSFT_s016
                                 + 0.0475660419529269 w_k1_MSFT_s017
                                 + 0.0121991634259386 w_k1_MSFT_s100
                                 + 0.0107994307566678 w_k1_MSFT_s101
                                 + 0.0369815451411289 w_k1_MSFT_s102
                                 + 0.0379844084084668 w_k1_MSFT_s103
                                 + 0.0153142202928836 w_k1_MSFT_s104
                                 + 0.00836777208000156 w_k1_MSFT_s105
                                 + 0.0111102193689871 w_k1_MSFT_s106
                                 + 0.00809090056390101 w_k1_MSFT_s107
                                 + 0.0876255269576193 w_k1_MSFT_s110
                                 + 0.0359364677298896 w_k1_MSFT_s111
                                 + 0.0236479590274505 w_k1_MSFT_s112
                                 + 0.0360445170908006 w_k1_MSFT_s113
                                 + 0.0267066848178578 w_k1_MSFT_s114
                                 + 0.0353087860490999 w_k1_MSFT_s115
                                 + 0.0279518987310686 w_k1_MSFT_s116
                                 + 0.0547649532472692 w_k1_MSFT_s117
                                 + 0.0280314116525195 w_k1_PG_s000
                                 + 0.0361987672250584 w_k1_PG_s001
                                 + 0.0469410728067616 w_k1_PG_s002
                                 + 0.0460818681000694 w_k1_PG_s003
                                 + 0.0198968457434967 w_k1_PG_s004
                                 + 0.0451734675732652 w_k1_PG_s005
                                 + 0.019039797321 w_k1_PG_s006
                                 + 0.0296798020152537 w_k1_PG_s007
                                 + 0.0198508201311067 w_k1_PG_s010
                                 + 0.0191111707978133 w_k1_PG_s011
                                 + 0.0348465098446387 w_k1_PG_s012
                                 + 0.0212919615618769 w_k1_PG_s013
                                 + 0.0412027459095917 w_k1_PG_s014
                                 + 0.0663844042305768 w_k1_PG_s015
                                 + 0.0177394078235755 w_k1_PG_s016
                                 + 0.0457746762557068 w_k1_PG_s017
                                 + 0.012587363628229 w_k1_PG_s100
                                 + 0.0108869956904797 w_k1_PG_s101
                                 + 0.0352205178472778 w_k1_PG_s102
                                 + 0.037882426497842 w_k1_PG_s103
                                 + 0.0158483811507445 w_k1_PG_s104
                                 + 0.00914418907927863 w_k1_PG_s105
                                 + 0.0103401826664154 w_k1_PG_s106
                                 + 0.0076538022322932 w_k1_PG_s107
                                 + 0.0833038663401321 w_k1_PG_s110
                                 + 0.0351179168236159 w_k1_PG_s111
                                 + 0.0260692272067173 w_k1_PG_s112
                                 + 0.0399566792878641 w_k1_PG_s113
                                 + 0.0271018857563425 w_k1_PG_s114
                                 + 0.0387001725399653 w_k1_PG_s115
                                 + 0.0285195807339242 w_k1_PG_s116
                                 + 0.0518625365487039 w_k1_PG_s117 >= 1.09
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
