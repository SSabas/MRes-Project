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
 w_k1_ADM_s0_return_constraint:  1.13618 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.942671 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.921659 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.08152 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.784494 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.08751 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.20424 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.817607 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.953897 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.777067 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.99068 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.21486 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.765973 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.996939 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.51112 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02473 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.02611 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.849478 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.87607 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.15886 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.915749 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.42691 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.07921 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.892356 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.17342 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.32755 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.755098 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.816389 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.00747 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.991499 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.930538 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.00826 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.18185 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.2454 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.661103 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.86553 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.918012 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.05971 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.1791 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.842578 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.846067 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.30067 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.781738 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16162 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.22051 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.09929 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.809644 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.76497 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.986162 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.55943 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.714767 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.41175 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.4826 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.05929 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.03936 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.7732 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.964104 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.26993 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.788119 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.26367 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.05478 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.10257 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.18933 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.70198 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.718716 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.788156 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.03261 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.764601 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.08689 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.10716 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.22443 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.47174 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.63907 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.752696 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.838322 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.06421 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26053 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.784919 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.812518 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.31623 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.693549 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.25686 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.20517 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.976874 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.930575 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.673127 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.876464 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.8813 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.63827 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.34958 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.46176 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.987343 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.06709 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.767403 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.726565 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.27448 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.702067 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.44604 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.14635 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.99681 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.20237 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.7397 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.721529 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.814064 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.07052 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.878009 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.12244 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.93882 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.32762 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.71756 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.64366 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.760203 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.979748 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.02582 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.14842 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.884787 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.893013 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.14267 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.785933 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.07961 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.20756 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.885558 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.996063 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.703099 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.898552 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.50209 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.775886 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.15468 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.61597 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.02487 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.987676 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.920397 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.799686 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.03881 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.741779 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.37951 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.06928 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.864303 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.25338 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.42097 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.693301 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.877137 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.962342 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.00359 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.943907 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.983623 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.28807 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.37725 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.728437 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.883115 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.00451 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.22669 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15803 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.908068 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.858592 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.25948 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.827815 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.12715 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.759001 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.61418 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.61897 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.03134 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.992032 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.13649 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.61076 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.36002 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.687369 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.978622 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.37272 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.65299 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.15123 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.94396 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.12195 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.02792 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.02215 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.67122 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.814456 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.46365 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.875109 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.82657 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.28565 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.543015 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.71715 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.01672 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.889569 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.29943 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.884291 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.672785 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.645836 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.596956 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11923 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.888434 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.970594 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.191 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.775498 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.97804 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.02173 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.01241 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.878664 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.67042 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.748979 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.33539 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.763283 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.9986 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.39193 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.02065 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.00374 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.754995 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.791337 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.12493 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.820579 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.20553 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.10595 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.1732 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09602 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.41986 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.818507 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.05795 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.15467 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.02555 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.12426 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.11651 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.21411 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.58726 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.879032 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.894665 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.04926 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.23735 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13856 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.901301 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.857638 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.26057 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.806805 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.16216 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.03304 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.26935 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.798824 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.921693 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.00341 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.36415 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.765281 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.32225 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.01123 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.956112 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.1397 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.816069 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.91533 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.50134 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.908409 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.19731 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.07194 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15583 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.95136 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.45788 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.806293 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.805632 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.20836 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.702772 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.26718 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.01407 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.03077 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.33109 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.753787 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.69241 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.861677 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.743093 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.32579 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.897576 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.682585 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.35744 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.900608 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.12365 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.12935 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.70927 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.717797 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.10919 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.20497 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.2062 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.657048 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.91392 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.7021 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.748331 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.21193 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.65376 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.11911 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.57384 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.757863 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.07363 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.99851 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.39634 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.856442 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.27878 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.680062 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.675044 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.89817 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.412674 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.82438 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.17769 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.981738 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.13874 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.733313 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.835233 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.700792 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.523368 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14895 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.878467 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.87956 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.16567 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.787016 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.11517 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.11689 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.04698 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.0507 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.729566 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.883558 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.44561 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.839449 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.09625 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.43438 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02878 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03804 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.789542 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.813591 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.10858 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.769184 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.2796 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.11678 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.00186 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.1932 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.41334 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.807531 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.895403 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.03827 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.07936 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.04624 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.967612 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.25184 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.33254 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.794243 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.88346 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.0116 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.06915 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10011 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.914805 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.914126 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.14596 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.835461 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.10828 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.08976 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.13258 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.959671 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.805777 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.945104 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.21324 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.823351 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.159 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.26524 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04501 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.08941 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.871324 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.883519 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.1807 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.872702 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.20166 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.04564 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.04668 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.10771 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.30818 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.841628 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.983408 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.11522 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.884647 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.00388 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.958099 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.13659 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.14196 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.739501 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.844213 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.9263 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.986988 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.353999451906157 z_k1_s0000
                                 - 1.36299785593818 z_k1_s0001
                                 - 0.691997811909191 z_k1_s0002
                                 - 0.610998664386832 z_k1_s0003
                                 - 0.311999572154468 z_k1_s0004
                                 - 0.827997319105343 z_k1_s0005
                                 - 0.588996348521672 z_k1_s0006
                                 - 0.68099665397661 z_k1_s0007
                                 - 0.122999977366905 z_k1_s0010
                                 - 0.412998559563105 z_k1_s0011
                                 - 0.330000129959841 z_k1_s0012
                                 - 0.513999223198815 z_k1_s0013
                                 - 0.250999822454518 z_k1_s0014
                                 - 0.261999218720318 z_k1_s0015
                                 - 0.200999842248451 z_k1_s0016
                                 - 0.512000422749642 z_k1_s0017
                                 - 0.262000312707787 z_k1_s0020
                                 - 0.183999815349214 z_k1_s0021
                                 - 0.241999552257237 z_k1_s0022
                                 - 0.206999904467801 z_k1_s0023
                                 - 0.174999809746272 z_k1_s0024
                                 - 0.633999197896844 z_k1_s0025
                                 - 0.290000928538818 z_k1_s0026
                                 - 0.249999407637716 z_k1_s0027
                                 - 0.235999855435379 z_k1_s0030
                                 - 0.350998956130228 z_k1_s0031
                                 - 0.660998812541079 z_k1_s0032
                                 - 0.226999833165402 z_k1_s0033
                                 - 0.264999745060967 z_k1_s0034
                                 - 0.455000670537986 z_k1_s0035
                                 - 0.355998513169373 z_k1_s0036
                                 - 0.180999810407689 z_k1_s0037
                                 - 0.803995783641552 z_k1_s0040
                                 - 0.632996162972396 z_k1_s0041
                                 - 0.68599766729401 z_k1_s0042
                                 - 0.403999387634251 z_k1_s0043
                                 - 1.05399786463878 z_k1_s0044
                                 - 0.345999343970151 z_k1_s0045
                                 - 0.59100173002685 z_k1_s0046
                                 - 1.26000055008286 z_k1_s0047
                                 - 0.197999841125711 z_k1_s0050
                                 - 0.370000635966179 z_k1_s0051
                                 - 0.155999927505046 z_k1_s0052
                                 - 0.169999898711934 z_k1_s0053
                                 - 0.218999037386907 z_k1_s0054
                                 - 0.194999800731431 z_k1_s0055
                                 - 0.342998968619136 z_k1_s0056
                                 - 0.521999640413171 z_k1_s0057
                                 - 0.173999678689908 z_k1_s0060
                                 - 0.307000529339613 z_k1_s0061
                                 - 0.718998891537369 z_k1_s0062
                                 - 0.300999224920714 z_k1_s0063
                                 - 0.590998250904375 z_k1_s0064
                                 - 0.191999908550645 z_k1_s0065
                                 - 0.177999846703277 z_k1_s0066
                                 - 0.168999731772909 z_k1_s0067
                                 - 0.778002643480906 z_k1_s0070
                                 - 0.559998441349703 z_k1_s0071
                                 - 0.753003133421157 z_k1_s0072
                                 - 0.428000557433662 z_k1_s0073
                                 - 0.48900171598228 z_k1_s0074
                                 - 0.470000675935172 z_k1_s0075
                                 - 0.603999462257126 z_k1_s0076
                                 - 0.625999972710838 z_k1_s0077
                                 - 0.228000571256776 z_k1_s0100
                                 - 0.136999988649381 z_k1_s0101
                                 - 0.20600071471475 z_k1_s0102
                                 - 0.116999825065285 z_k1_s0103
                                 - 0.0530000561972953 z_k1_s0104
                                 - 0.16800008050449 z_k1_s0105
                                 - 0.047000114922226 z_k1_s0106
                                 - 0.122000674462505 z_k1_s0107
                                 - 0.265000340729545 z_k1_s0110
                                 - 1.01400233940067 z_k1_s0111
                                 - 0.876000132244041 z_k1_s0112
                                 - 0.406000857079494 z_k1_s0113
                                 - 0.357000222266028 z_k1_s0114
                                 - 0.276000378410035 z_k1_s0115
                                 - 0.263000367233119 z_k1_s0116
                                 - 0.212000125823519 z_k1_s0117
                                 - 0.529002736031762 z_k1_s0120
                                 - 0.230000738273309 z_k1_s0121
                                 - 0.131000409212155 z_k1_s0122
                                 - 0.444000657968754 z_k1_s0123
                                 - 0.441000828861389 z_k1_s0124
                                 - 0.371000491343083 z_k1_s0125
                                 - 0.267000360602725 z_k1_s0126
                                 - 0.182000618046808 z_k1_s0127
                                 - 0.42100085687676 z_k1_s0130
                                 - 0.360000850833064 z_k1_s0131
                                 - 0.78700186814063 z_k1_s0132
                                 - 0.352000474425041 z_k1_s0133
                                 - 0.487000671612229 z_k1_s0134
                                 - 1.44600063875798 z_k1_s0135
                                 - 0.640000862590985 z_k1_s0136
                                 - 0.382000789078221 z_k1_s0137
                                 - 0.163000664114589 z_k1_s0140
                                 - 0.159999922957854 z_k1_s0141
                                 - 0.187000273357616 z_k1_s0142
                                 - 0.0810001663977664 z_k1_s0143
                                 - 0.320001109917879 z_k1_s0144
                                 - 0.107000185478472 z_k1_s0145
                                 - 0.159000097239666 z_k1_s0146
                                 - 0.0870001319086316 z_k1_s0147
                                 - 0.220000747479394 z_k1_s0150
                                 - 0.151000107551376 z_k1_s0151
                                 - 0.191999992563201 z_k1_s0152
                                 - 0.220000747479394 z_k1_s0153
                                 - 0.245000179510516 z_k1_s0154
                                 - 0.134000070409807 z_k1_s0155
                                 - 0.250999801277753 z_k1_s0156
                                 - 0.0990001095655095 z_k1_s0157
                                 - 0.299999399769745 z_k1_s0160
                                 - 0.327998963114225 z_k1_s0161
                                 - 0.279999667338048 z_k1_s0162
                                 - 1.01899867076938 z_k1_s0163
                                 - 0.28099945226497 z_k1_s0164
                                 - 0.257999434154064 z_k1_s0165
                                 - 0.356000388255589 z_k1_s0166
                                 - 0.28099945226497 z_k1_s0167
                                 - 0.430000285954332 z_k1_s0170
                                 - 0.266999980487983 z_k1_s0171
                                 - 0.493000219461628 z_k1_s0172
                                 - 0.310000009083416 z_k1_s0173
                                 - 0.388999697040255 z_k1_s0174
                                 - 0.724001387987943 z_k1_s0175
                                 - 0.772998040827971 z_k1_s0176
                                 - 0.851001186693148 z_k1_s0177
                                 - 0.268000486285766 z_k1_s1000
                                 - 0.0879998743269045 z_k1_s1001
                                 - 0.101999993996979 z_k1_s1002
                                 - 0.313000383225692 z_k1_s1003
                                 - 0.173999487701142 z_k1_s1004
                                 - 0.432998970399766 z_k1_s1005
                                 - 0.180999888935898 z_k1_s1006
                                 - 0.147999850713379 z_k1_s1007
                                 - 0.318998423374139 z_k1_s1010
                                 - 0.133999409218009 z_k1_s1011
                                 - 0.106999599862605 z_k1_s1012
                                 - 0.295998108051049 z_k1_s1013
                                 - 0.110999565793058 z_k1_s1014
                                 - 0.129999443287556 z_k1_s1015
                                 - 0.285998586473366 z_k1_s1016
                                 - 0.189998774944972 z_k1_s1017
                                 - 0.428999942670371 z_k1_s1020
                                 - 0.158999580849938 z_k1_s1021
                                 - 0.243999428051338 z_k1_s1022
                                 - 0.162999579214699 z_k1_s1023
                                 - 0.430999705753214 z_k1_s1024
                                 - 0.321999632263712 z_k1_s1025
                                 - 0.17599969194994 z_k1_s1026
                                 - 0.436998995001743 z_k1_s1027
                                 - 0.13799982835456 z_k1_s1030
                                 - 0.13799982835456 z_k1_s1031
                                 - 0.131999632165496 z_k1_s1032
                                 - 0.1089998169062 z_k1_s1033
                                 - 0.212999938720903 z_k1_s1034
                                 - 0.231999388987795 z_k1_s1035
                                 - 0.0869999172452327 z_k1_s1036
                                 - 0.121999305183723 z_k1_s1037
                                 - 0.290999262017533 z_k1_s1040
                                 - 0.236998776944821 z_k1_s1041
                                 - 0.448998446647414 z_k1_s1042
                                 - 0.0939995594580895 z_k1_s1043
                                 - 0.0999996133550576 z_k1_s1044
                                 - 0.17499877867372 z_k1_s1045
                                 - 0.0999996133550576 z_k1_s1046
                                 - 0.229999831393497 z_k1_s1047
                                 - 0.153000442916122 z_k1_s1050
                                 - 0.448999317603692 z_k1_s1051
                                 - 0.0959999521905359 z_k1_s1052
                                 - 0.135999919569927 z_k1_s1053
                                 - 0.278000436226711 z_k1_s1054
                                 - 0.11300001833677 z_k1_s1055
                                 - 0.207000330298306 z_k1_s1056
                                 - 0.0920000621325877 z_k1_s1057
                                 - 0.248000886413393 z_k1_s1060
                                 - 0.176000439002506 z_k1_s1061
                                 - 0.307000598373207 z_k1_s1062
                                 - 0.149000271223423 z_k1_s1063
                                 - 0.132000329251879 z_k1_s1064
                                 - 0.12600034106778 z_k1_s1065
                                 - 0.240001000368211 z_k1_s1066
                                 - 0.095000230266106 z_k1_s1067
                                 - 0.121999987450737 z_k1_s1070
                                 - 0.252000085921113 z_k1_s1071
                                 - 0.296000270129439 z_k1_s1072
                                 - 0.264998880529029 z_k1_s1073
                                 - 0.325999620453858 z_k1_s1074
                                 - 0.380998784715036 z_k1_s1075
                                 - 0.299999899239567 z_k1_s1076
                                 - 0.189999865118444 z_k1_s1077
                                 - 1.43900353448239 z_k1_s1100
                                 - 0.443001115011731 z_k1_s1101
                                 - 0.556001046980549 z_k1_s1102
                                 - 0.345000510352376 z_k1_s1103
                                 - 0.966003150150394 z_k1_s1104
                                 - 0.355000471692908 z_k1_s1105
                                 - 0.478000979703595 z_k1_s1106
                                 - 1.56500501441737 z_k1_s1107
                                 - 0.264000992783056 z_k1_s1110
                                 - 1.06900123169652 z_k1_s1111
                                 - 0.818000897987484 z_k1_s1112
                                 - 0.658001779286842 z_k1_s1113
                                 - 1.30700566424515 z_k1_s1114
                                 - 1.01700403301601 z_k1_s1115
                                 - 1.19100093354183 z_k1_s1116
                                 - 0.473001183997109 z_k1_s1117
                                 - 0.252000153926419 z_k1_s1120
                                 - 0.768001807281721 z_k1_s1121
                                 - 0.580001934062229 z_k1_s1122
                                 - 0.194000350820548 z_k1_s1123
                                 - 0.568999367158771 z_k1_s1124
                                 - 0.292000085430529 z_k1_s1125
                                 - 0.563999619658477 z_k1_s1126
                                 - 0.684001365672563 z_k1_s1127
                                 - 0.319000245756833 z_k1_s1130
                                 - 0.317000146836683 z_k1_s1131
                                 - 0.360999146078529 z_k1_s1132
                                 - 0.309000104156244 z_k1_s1133
                                 - 0.706000324797154 z_k1_s1134
                                 - 0.773998746079992 z_k1_s1135
                                 - 0.239000171952606 z_k1_s1136
                                 - 0.50499850232578 z_k1_s1137
                                 - 0.74700080334004 z_k1_s1140
                                 - 0.457001117399646 z_k1_s1141
                                 - 1.07900231616089 z_k1_s1142
                                 - 0.295000980779014 z_k1_s1143
                                 - 0.320000693654222 z_k1_s1144
                                 - 0.400001127118466 z_k1_s1145
                                 - 0.870004778936324 z_k1_s1146
                                 - 1.03300455040303 z_k1_s1147
                                 - 0.483001055477867 z_k1_s1150
                                 - 0.96300359716472 z_k1_s1151
                                 - 0.648001728697827 z_k1_s1152
                                 - 0.58399954740146 z_k1_s1153
                                 - 0.444001129644665 z_k1_s1154
                                 - 0.766002820299141 z_k1_s1155
                                 - 0.630003736933786 z_k1_s1156
                                 - 0.417001180990909 z_k1_s1157
                                 - 0.150999969291642 z_k1_s1160
                                 - 0.321998942973034 z_k1_s1161
                                 - 0.140999956096833 z_k1_s1162
                                 - 0.428000769137786 z_k1_s1163
                                 - 0.212999846699482 z_k1_s1164
                                 - 0.459999584961334 z_k1_s1165
                                 - 0.693000224550334 z_k1_s1166
                                 - 0.146999861813731 z_k1_s1167
                                 - 0.236999880490813 z_k1_s1170
                                 - 0.745999366000642 z_k1_s1171
                                 - 0.206999963466652 z_k1_s1172
                                 - 0.224999913681148 z_k1_s1173
                                 - 0.299999170241602 z_k1_s1174
                                 - 0.315998768654523 z_k1_s1175
                                 - 0.461999794171961 z_k1_s1176
                                 - 0.186999929450554 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.813389 w_k1_ADM_s000
                                 + 0.562923 w_k1_AXP_s000
                                 + 0.589195 w_k1_DIS_s000
                                 + 0.733955 w_k1_FDX_s000
                                 + 0.680785 w_k1_IBM_s000
                                 + 0.733165 w_k1_INTC_s000
                                 + 0.686879 w_k1_KO_s000
                                 + 0.641436 w_k1_MAT_s000
                                 + 0.803444 w_k1_MSFT_s000
                                 + 0.856336 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.823017 w_k1_ADM_s000
                                 + 0.848939 w_k1_AXP_s000
                                 + 0.765378 w_k1_DIS_s000
                                 + 0.861726 w_k1_FDX_s000
                                 + 0.98557 w_k1_IBM_s000
                                 + 1.00908 w_k1_INTC_s000
                                 + 0.807562 w_k1_KO_s000
                                 + 0.724866 w_k1_MAT_s000
                                 + 0.860077 w_k1_MSFT_s000
                                 + 0.795498 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.881994 w_k1_ADM_s000
                                 + 0.887222 w_k1_AXP_s000
                                 + 1.24773 w_k1_DIS_s000
                                 + 0.993897 w_k1_FDX_s000
                                 + 1.20684 w_k1_IBM_s000
                                 + 1.14586 w_k1_INTC_s000
                                 + 1.24549 w_k1_KO_s000 + 1.13612 w_k1_MAT_s000
                                 + 1.17325 w_k1_MSFT_s000
                                 + 0.905781 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.19064 w_k1_ADM_s000
                                 + 1.61795 w_k1_AXP_s000
                                 + 1.47724 w_k1_DIS_s000
                                 + 1.38958 w_k1_FDX_s000
                                 + 1.29615 w_k1_IBM_s000
                                 + 1.36151 w_k1_INTC_s000
                                 + 1.27027 w_k1_KO_s000 + 1.40767 w_k1_MAT_s000
                                 + 1.34462 w_k1_MSFT_s000 + 1.1658 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.1338 w_k1_ADM_s000
                                 + 1.11036 w_k1_AXP_s000
                                 + 0.96878 w_k1_DIS_s000
                                 + 1.05541 w_k1_FDX_s000
                                 + 1.04142 w_k1_IBM_s000
                                 + 1.00586 w_k1_INTC_s000
                                 + 1.11227 w_k1_KO_s000 + 1.09184 w_k1_MAT_s000
                                 + 0.98424 w_k1_MSFT_s000
                                 + 0.922442 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.926087 w_k1_ADM_s000
                                 + 1.15072 w_k1_AXP_s000 + 1.0539 w_k1_DIS_s000
                                 + 0.93619 w_k1_FDX_s000
                                 + 1.38791 w_k1_IBM_s000
                                 + 1.02212 w_k1_INTC_s000
                                 + 1.12041 w_k1_KO_s000
                                 + 0.958077 w_k1_MAT_s000
                                 + 1.06797 w_k1_MSFT_s000
                                 + 0.961329 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.985837 w_k1_ADM_s000
                                 + 0.832724 w_k1_AXP_s000
                                 + 0.907424 w_k1_DIS_s000
                                 + 1.05855 w_k1_FDX_s000
                                 + 0.682502 w_k1_IBM_s000
                                 + 0.936204 w_k1_INTC_s000
                                 + 0.718116 w_k1_KO_s000
                                 + 0.533059 w_k1_MAT_s000
                                 + 1.03484 w_k1_MSFT_s000
                                 + 0.859692 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.04287 w_k1_ADM_s000
                                 + 1.08208 w_k1_AXP_s000
                                 + 1.44263 w_k1_DIS_s000
                                 + 1.29703 w_k1_FDX_s000
                                 + 1.05888 w_k1_IBM_s000
                                 + 1.3912 w_k1_INTC_s000 + 1.07808 w_k1_KO_s000
                                 + 0.888108 w_k1_MAT_s000
                                 + 1.33737 w_k1_MSFT_s000 + 1.0565 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.87002 w_k1_ADM_s001
                                 + 1.59358 w_k1_AXP_s001 + 1.379 w_k1_DIS_s001
                                 + 1.41469 w_k1_FDX_s001
                                 + 1.07214 w_k1_IBM_s001
                                 + 1.25307 w_k1_INTC_s001
                                 + 1.20873 w_k1_KO_s001 + 1.40141 w_k1_MAT_s001
                                 + 1.21458 w_k1_MSFT_s001
                                 + 1.16044 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.1022 w_k1_ADM_s001
                                 + 1.0362 w_k1_AXP_s001 + 1.03394 w_k1_DIS_s001
                                 + 0.995975 w_k1_FDX_s001
                                 + 1.31652 w_k1_IBM_s001
                                 + 0.988744 w_k1_INTC_s001
                                 + 1.07108 w_k1_KO_s001 + 1.32126 w_k1_MAT_s001
                                 + 1.00225 w_k1_MSFT_s001
                                 + 0.963352 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.3649 w_k1_ADM_s001
                                 + 1.21276 w_k1_AXP_s001
                                 + 1.56479 w_k1_DIS_s001
                                 + 1.24174 w_k1_FDX_s001
                                 + 1.55503 w_k1_IBM_s001
                                 + 1.06345 w_k1_INTC_s001
                                 + 1.35426 w_k1_KO_s001 + 1.27614 w_k1_MAT_s001
                                 + 1.06184 w_k1_MSFT_s001 + 1.1401 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.41515 w_k1_ADM_s001
                                 + 1.11714 w_k1_AXP_s001
                                 + 1.12366 w_k1_DIS_s001
                                 + 1.19848 w_k1_FDX_s001
                                 + 0.790707 w_k1_IBM_s001
                                 + 0.982412 w_k1_INTC_s001
                                 + 0.95364 w_k1_KO_s001
                                 + 0.673516 w_k1_MAT_s001
                                 + 1.0154 w_k1_MSFT_s001
                                 + 0.969951 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.04467 w_k1_ADM_s001
                                 + 0.734681 w_k1_AXP_s001
                                 + 0.850437 w_k1_DIS_s001
                                 + 0.808412 w_k1_FDX_s001
                                 + 1.0941 w_k1_IBM_s001
                                 + 0.680634 w_k1_INTC_s001
                                 + 0.924991 w_k1_KO_s001
                                 + 0.91814 w_k1_MAT_s001
                                 + 0.755366 w_k1_MSFT_s001
                                 + 0.831232 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.04814 w_k1_ADM_s001
                                 + 0.671639 w_k1_AXP_s001
                                 + 0.753282 w_k1_DIS_s001
                                 + 0.876235 w_k1_FDX_s001
                                 + 0.591938 w_k1_IBM_s001
                                 + 0.80046 w_k1_INTC_s001
                                 + 0.660343 w_k1_KO_s001
                                 + 0.642999 w_k1_MAT_s001
                                 + 0.83605 w_k1_MSFT_s001
                                 + 0.793256 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.757631 w_k1_ADM_s001
                                 + 0.615764 w_k1_AXP_s001
                                 + 0.676045 w_k1_DIS_s001
                                 + 0.702523 w_k1_FDX_s001
                                 + 0.786639 w_k1_IBM_s001
                                 + 0.72325 w_k1_INTC_s001
                                 + 0.707509 w_k1_KO_s001
                                 + 0.694389 w_k1_MAT_s001
                                 + 0.747193 w_k1_MSFT_s001
                                 + 0.747307 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.946499 w_k1_ADM_s001
                                 + 0.992215 w_k1_AXP_s001
                                 + 0.953313 w_k1_DIS_s001
                                 + 0.950501 w_k1_FDX_s001
                                 + 0.924348 w_k1_IBM_s001
                                 + 0.861859 w_k1_INTC_s001
                                 + 0.947481 w_k1_KO_s001
                                 + 0.729467 w_k1_MAT_s001
                                 + 0.884724 w_k1_MSFT_s001
                                 + 0.909859 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.940041 w_k1_ADM_s002
                                 + 0.81248 w_k1_AXP_s002
                                 + 0.668924 w_k1_DIS_s002
                                 + 0.850052 w_k1_FDX_s002
                                 + 0.691547 w_k1_IBM_s002
                                 + 0.850758 w_k1_INTC_s002
                                 + 0.850176 w_k1_KO_s002
                                 + 0.767991 w_k1_MAT_s002
                                 + 0.86044 w_k1_MSFT_s002
                                 + 0.744131 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.826974 w_k1_ADM_s002
                                 + 0.951512 w_k1_AXP_s002
                                 + 0.738952 w_k1_DIS_s002
                                 + 0.780011 w_k1_FDX_s002
                                 + 1.0818 w_k1_IBM_s002
                                 + 0.870587 w_k1_INTC_s002
                                 + 1.05975 w_k1_KO_s002 + 1.02928 w_k1_MAT_s002
                                 + 0.782836 w_k1_MSFT_s002
                                 + 0.986905 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.24394 w_k1_ADM_s002
                                 + 0.943397 w_k1_AXP_s002
                                 + 1.06704 w_k1_DIS_s002
                                 + 1.15577 w_k1_FDX_s002
                                 + 0.699767 w_k1_IBM_s002
                                 + 1.09749 w_k1_INTC_s002
                                 + 0.968573 w_k1_KO_s002
                                 + 0.526631 w_k1_MAT_s002
                                 + 0.978813 w_k1_MSFT_s002
                                 + 0.974756 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.899716 w_k1_ADM_s002
                                 + 0.881871 w_k1_AXP_s002
                                 + 0.715466 w_k1_DIS_s002
                                 + 0.884709 w_k1_FDX_s002
                                 + 1.01051 w_k1_IBM_s002
                                 + 0.879872 w_k1_INTC_s002
                                 + 0.97653 w_k1_KO_s002 + 1.07874 w_k1_MAT_s002
                                 + 0.847649 w_k1_MSFT_s002
                                 + 0.896463 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.780351 w_k1_ADM_s002
                                 + 0.928054 w_k1_AXP_s002
                                 + 0.979336 w_k1_DIS_s002
                                 + 0.950806 w_k1_FDX_s002
                                 + 0.785707 w_k1_IBM_s002
                                 + 1.08076 w_k1_INTC_s002
                                 + 1.01301 w_k1_KO_s002
                                 + 0.728403 w_k1_MAT_s002
                                 + 0.965352 w_k1_MSFT_s002
                                 + 0.904743 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.927941 w_k1_ADM_s002
                                 + 1.15971 w_k1_AXP_s002
                                 + 1.25482 w_k1_DIS_s002
                                 + 1.04129 w_k1_FDX_s002
                                 + 1.19203 w_k1_IBM_s002
                                 + 1.12245 w_k1_INTC_s002
                                 + 1.16042 w_k1_KO_s002
                                 + 0.897473 w_k1_MAT_s002
                                 + 1.01799 w_k1_MSFT_s002
                                 + 1.10948 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.82299 w_k1_ADM_s002
                                 + 1.12899 w_k1_AXP_s002
                                 + 1.16871 w_k1_DIS_s002
                                 + 1.02632 w_k1_FDX_s002
                                 + 1.66252 w_k1_IBM_s002
                                 + 1.19029 w_k1_INTC_s002
                                 + 1.33662 w_k1_KO_s002 + 1.50231 w_k1_MAT_s002
                                 + 1.05622 w_k1_MSFT_s002 + 1.1132 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.36589 w_k1_ADM_s002
                                 + 1.6578 w_k1_AXP_s002
                                 + 0.935839 w_k1_DIS_s002
                                 + 1.2014 w_k1_FDX_s002
                                 + 0.969922 w_k1_IBM_s002
                                 + 1.16792 w_k1_INTC_s002
                                 + 1.02037 w_k1_KO_s002 + 1.05657 w_k1_MAT_s002
                                 + 0.856361 w_k1_MSFT_s002 + 1.005 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.867223 w_k1_ADM_s003
                                 + 0.831163 w_k1_AXP_s003
                                 + 0.654903 w_k1_DIS_s003
                                 + 0.755865 w_k1_FDX_s003
                                 + 1.4267 w_k1_IBM_s003
                                 + 0.844063 w_k1_INTC_s003
                                 + 0.901934 w_k1_KO_s003
                                 + 1.20115 w_k1_MAT_s003
                                 + 0.814273 w_k1_MSFT_s003
                                 + 0.934945 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.934536 w_k1_ADM_s003
                                 + 0.992009 w_k1_AXP_s003
                                 + 0.98611 w_k1_DIS_s003
                                 + 1.01835 w_k1_FDX_s003
                                 + 0.669754 w_k1_IBM_s003
                                 + 1.14905 w_k1_INTC_s003
                                 + 0.832023 w_k1_KO_s003
                                 + 0.540052 w_k1_MAT_s003
                                 + 1.07112 w_k1_MSFT_s003
                                 + 0.936279 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.889654 w_k1_ADM_s003
                                 + 0.917335 w_k1_AXP_s003
                                 + 0.932605 w_k1_DIS_s003
                                 + 1.03839 w_k1_FDX_s003
                                 + 0.945857 w_k1_IBM_s003
                                 + 1.1912 w_k1_INTC_s003
                                 + 0.888597 w_k1_KO_s003
                                 + 0.854861 w_k1_MAT_s003
                                 + 1.04396 w_k1_MSFT_s003
                                 + 1.00672 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.08052 w_k1_ADM_s003
                                 + 1.4711 w_k1_AXP_s003 + 1.73029 w_k1_DIS_s003
                                 + 1.25119 w_k1_FDX_s003
                                 + 1.49706 w_k1_IBM_s003
                                 + 1.44263 w_k1_INTC_s003
                                 + 1.28784 w_k1_KO_s003 + 1.25532 w_k1_MAT_s003
                                 + 1.21226 w_k1_MSFT_s003
                                 + 1.17928 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.23876 w_k1_ADM_s003
                                 + 1.17665 w_k1_AXP_s003 + 1.5989 w_k1_DIS_s003
                                 + 1.28978 w_k1_FDX_s003
                                 + 1.03241 w_k1_IBM_s003
                                 + 1.2568 w_k1_INTC_s003 + 1.13315 w_k1_KO_s003
                                 + 0.795224 w_k1_MAT_s003
                                 + 1.30416 w_k1_MSFT_s003
                                 + 1.30153 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.850885 w_k1_ADM_s003
                                 + 0.926685 w_k1_AXP_s003
                                 + 1.11002 w_k1_DIS_s003
                                 + 1.00385 w_k1_FDX_s003
                                 + 1.27352 w_k1_IBM_s003
                                 + 1.36193 w_k1_INTC_s003
                                 + 1.10043 w_k1_KO_s003
                                 + 0.940646 w_k1_MAT_s003
                                 + 1.08124 w_k1_MSFT_s003
                                 + 1.16226 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.709328 w_k1_ADM_s003
                                 + 0.739102 w_k1_AXP_s003
                                 + 0.691348 w_k1_DIS_s003
                                 + 0.783881 w_k1_FDX_s003
                                 + 0.82868 w_k1_IBM_s003
                                 + 1.04742 w_k1_INTC_s003
                                 + 0.839893 w_k1_KO_s003
                                 + 0.773146 w_k1_MAT_s003
                                 + 0.920579 w_k1_MSFT_s003
                                 + 0.882755 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.769976 w_k1_ADM_s003
                                 + 0.635911 w_k1_AXP_s003
                                 + 0.568808 w_k1_DIS_s003
                                 + 0.769965 w_k1_FDX_s003
                                 + 0.595051 w_k1_IBM_s003
                                 + 0.748765 w_k1_INTC_s003
                                 + 0.622099 w_k1_KO_s003
                                 + 0.529679 w_k1_MAT_s003
                                 + 0.708939 w_k1_MSFT_s003
                                 + 0.756374 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.06577 w_k1_ADM_s004
                                 + 1.01141 w_k1_AXP_s004
                                 + 1.32586 w_k1_DIS_s004
                                 + 1.13578 w_k1_FDX_s004
                                 + 0.889403 w_k1_IBM_s004
                                 + 1.2032 w_k1_INTC_s004 + 1.10377 w_k1_KO_s004
                                 + 0.887345 w_k1_MAT_s004
                                 + 1.26953 w_k1_MSFT_s004
                                 + 1.11615 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.757425 w_k1_ADM_s004
                                 + 0.683363 w_k1_AXP_s004
                                 + 0.655644 w_k1_DIS_s004
                                 + 0.772958 w_k1_FDX_s004
                                 + 0.858186 w_k1_IBM_s004
                                 + 1.07734 w_k1_INTC_s004
                                 + 0.783635 w_k1_KO_s004
                                 + 0.771028 w_k1_MAT_s004
                                 + 0.848313 w_k1_MSFT_s004
                                 + 0.883911 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.983449 w_k1_ADM_s004
                                 + 1.40528 w_k1_AXP_s004
                                 + 1.39616 w_k1_DIS_s004
                                 + 1.14168 w_k1_FDX_s004
                                 + 1.44604 w_k1_IBM_s004
                                 + 1.28105 w_k1_INTC_s004
                                 + 1.23874 w_k1_KO_s004 + 1.18936 w_k1_MAT_s004
                                 + 1.13295 w_k1_MSFT_s004
                                 + 1.20699 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.26223 w_k1_ADM_s004
                                 + 1.27269 w_k1_AXP_s004
                                 + 0.819905 w_k1_DIS_s004
                                 + 0.990316 w_k1_FDX_s004
                                 + 0.842674 w_k1_IBM_s004
                                 + 1.21872 w_k1_INTC_s004
                                 + 0.825035 w_k1_KO_s004
                                 + 0.87457 w_k1_MAT_s004
                                 + 0.96327 w_k1_MSFT_s004 + 1.0465 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.799379 w_k1_ADM_s004
                                 + 0.841487 w_k1_AXP_s004
                                 + 0.982681 w_k1_DIS_s004
                                 + 1.04602 w_k1_FDX_s004
                                 + 0.847948 w_k1_IBM_s004
                                 + 1.20124 w_k1_INTC_s004
                                 + 0.834437 w_k1_KO_s004
                                 + 0.702743 w_k1_MAT_s004
                                 + 1.03788 w_k1_MSFT_s004
                                 + 0.990848 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.922381 w_k1_ADM_s004
                                 + 0.716942 w_k1_AXP_s004
                                 + 0.737869 w_k1_DIS_s004
                                 + 0.874404 w_k1_FDX_s004 + 0.676 w_k1_IBM_s004
                                 + 0.866954 w_k1_INTC_s004
                                 + 0.720508 w_k1_KO_s004
                                 + 0.652624 w_k1_MAT_s004
                                 + 0.840044 w_k1_MSFT_s004
                                 + 0.876847 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.2042 w_k1_ADM_s004
                                 + 1.50266 w_k1_AXP_s004
                                 + 1.65805 w_k1_DIS_s004
                                 + 1.38599 w_k1_FDX_s004
                                 + 1.15758 w_k1_IBM_s004
                                 + 1.62952 w_k1_INTC_s004
                                 + 1.22604 w_k1_KO_s004
                                 + 0.969819 w_k1_MAT_s004
                                 + 1.39049 w_k1_MSFT_s004
                                 + 1.12907 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.785957 w_k1_ADM_s004
                                 + 1.03864 w_k1_AXP_s004
                                 + 0.924527 w_k1_DIS_s004
                                 + 0.94152 w_k1_FDX_s004
                                 + 1.21159 w_k1_IBM_s004
                                 + 1.17806 w_k1_INTC_s004
                                 + 1.09388 w_k1_KO_s004 + 1.12339 w_k1_MAT_s004
                                 + 0.954966 w_k1_MSFT_s004
                                 + 1.0703 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.989372 w_k1_ADM_s005
                                 + 0.702363 w_k1_AXP_s005
                                 + 0.901839 w_k1_DIS_s005
                                 + 0.870149 w_k1_FDX_s005
                                 + 0.694921 w_k1_IBM_s005
                                 + 0.958791 w_k1_INTC_s005
                                 + 0.849361 w_k1_KO_s005
                                 + 0.651402 w_k1_MAT_s005
                                 + 1.02914 w_k1_MSFT_s005
                                 + 0.975734 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.31179 w_k1_ADM_s005
                                 + 1.27452 w_k1_AXP_s005
                                 + 1.27028 w_k1_DIS_s005
                                 + 1.30457 w_k1_FDX_s005
                                 + 0.914252 w_k1_IBM_s005
                                 + 1.29813 w_k1_INTC_s005
                                 + 1.17309 w_k1_KO_s005
                                 + 0.962923 w_k1_MAT_s005
                                 + 1.27921 w_k1_MSFT_s005
                                 + 1.06876 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.780127 w_k1_ADM_s005
                                 + 0.688753 w_k1_AXP_s005
                                 + 0.69362 w_k1_DIS_s005
                                 + 0.706138 w_k1_FDX_s005
                                 + 1.00546 w_k1_IBM_s005
                                 + 0.8195 w_k1_INTC_s005
                                 + 0.936081 w_k1_KO_s005
                                 + 0.956318 w_k1_MAT_s005
                                 + 0.852469 w_k1_MSFT_s005
                                 + 0.793645 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.859997 w_k1_ADM_s005
                                 + 0.749141 w_k1_AXP_s005
                                 + 0.99337 w_k1_DIS_s005
                                 + 0.851665 w_k1_FDX_s005
                                 + 1.20941 w_k1_IBM_s005
                                 + 0.925474 w_k1_INTC_s005
                                 + 1.17118 w_k1_KO_s005 + 1.34328 w_k1_MAT_s005
                                 + 1.12964 w_k1_MSFT_s005
                                 + 1.05843 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.03225 w_k1_ADM_s005
                                 + 0.671877 w_k1_AXP_s005
                                 + 0.670112 w_k1_DIS_s005
                                 + 0.816508 w_k1_FDX_s005
                                 + 0.615272 w_k1_IBM_s005
                                 + 0.883267 w_k1_INTC_s005
                                 + 0.65705 w_k1_KO_s005
                                 + 0.563508 w_k1_MAT_s005
                                 + 0.840127 w_k1_MSFT_s005
                                 + 0.851443 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.19669 w_k1_ADM_s005
                                 + 1.19633 w_k1_AXP_s005
                                 + 1.08731 w_k1_DIS_s005
                                 + 1.03597 w_k1_FDX_s005
                                 + 1.62859 w_k1_IBM_s005
                                 + 1.23865 w_k1_INTC_s005
                                 + 1.28927 w_k1_KO_s005 + 1.46396 w_k1_MAT_s005
                                 + 1.07485 w_k1_MSFT_s005
                                 + 1.08504 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.26885 w_k1_ADM_s005
                                 + 1.20527 w_k1_AXP_s005
                                 + 1.51156 w_k1_DIS_s005
                                 + 1.21371 w_k1_FDX_s005
                                 + 1.25266 w_k1_IBM_s005
                                 + 1.20076 w_k1_INTC_s005
                                 + 1.32054 w_k1_KO_s005 + 1.21134 w_k1_MAT_s005
                                 + 1.18731 w_k1_MSFT_s005
                                 + 1.25619 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.1475 w_k1_ADM_s005
                                 + 0.882221 w_k1_AXP_s005
                                 + 0.932214 w_k1_DIS_s005
                                 + 0.964816 w_k1_FDX_s005
                                 + 0.919687 w_k1_IBM_s005
                                 + 0.926762 w_k1_INTC_s005
                                 + 1.02054 w_k1_KO_s005
                                 + 0.963527 w_k1_MAT_s005
                                 + 0.907351 w_k1_MSFT_s005
                                 + 0.940488 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.47936 w_k1_ADM_s006
                                 + 1.0083 w_k1_AXP_s006
                                 + 0.949389 w_k1_DIS_s006 + 1.214 w_k1_FDX_s006
                                 + 0.72339 w_k1_IBM_s006
                                 + 1.00025 w_k1_INTC_s006
                                 + 0.726672 w_k1_KO_s006
                                 + 0.685621 w_k1_MAT_s006
                                 + 0.930911 w_k1_MSFT_s006
                                 + 0.800159 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.792162 w_k1_ADM_s006
                                 + 0.741967 w_k1_AXP_s006
                                 + 0.690043 w_k1_DIS_s006
                                 + 0.806216 w_k1_FDX_s006
                                 + 1.02166 w_k1_IBM_s006
                                 + 0.854886 w_k1_INTC_s006
                                 + 0.84339 w_k1_KO_s006 + 1.02588 w_k1_MAT_s006
                                 + 0.798733 w_k1_MSFT_s006
                                 + 0.885436 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.11757 w_k1_ADM_s006
                                 + 1.09438 w_k1_AXP_s006
                                 + 1.21088 w_k1_DIS_s006
                                 + 1.15994 w_k1_FDX_s006
                                 + 1.12868 w_k1_IBM_s006
                                 + 0.985734 w_k1_INTC_s006
                                 + 1.00975 w_k1_KO_s006 + 1.2127 w_k1_MAT_s006
                                 + 1.02424 w_k1_MSFT_s006
                                 + 1.07089 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.926558 w_k1_ADM_s006
                                 + 0.594851 w_k1_AXP_s006
                                 + 0.764138 w_k1_DIS_s006
                                 + 0.890198 w_k1_FDX_s006
                                 + 0.717998 w_k1_IBM_s006
                                 + 0.882966 w_k1_INTC_s006
                                 + 0.758855 w_k1_KO_s006
                                 + 0.543899 w_k1_MAT_s006
                                 + 0.854762 w_k1_MSFT_s006
                                 + 0.837007 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.14879 w_k1_ADM_s006
                                 + 0.948652 w_k1_AXP_s006
                                 + 0.893895 w_k1_DIS_s006
                                 + 0.935368 w_k1_FDX_s006
                                 + 1.27076 w_k1_IBM_s006
                                 + 1.09754 w_k1_INTC_s006
                                 + 1.08488 w_k1_KO_s006
                                 + 0.849346 w_k1_MAT_s006
                                 + 0.887371 w_k1_MSFT_s006
                                 + 1.05916 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.00397 w_k1_ADM_s006
                                 + 1.56304 w_k1_AXP_s006
                                 + 0.903471 w_k1_DIS_s006
                                 + 1.22103 w_k1_FDX_s006
                                 + 0.936932 w_k1_IBM_s006
                                 + 1.15435 w_k1_INTC_s006
                                 + 0.717131 w_k1_KO_s006
                                 + 1.18056 w_k1_MAT_s006
                                 + 0.999325 w_k1_MSFT_s006
                                 + 1.03114 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.553482 w_k1_ADM_s006
                                 + 0.571256 w_k1_AXP_s006
                                 + 0.591434 w_k1_DIS_s006
                                 + 0.665657 w_k1_FDX_s006
                                 + 0.721309 w_k1_IBM_s006
                                 + 0.749971 w_k1_INTC_s006
                                 + 0.651952 w_k1_KO_s006
                                 + 0.605354 w_k1_MAT_s006
                                 + 0.650884 w_k1_MSFT_s006
                                 + 0.743854 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.44276 w_k1_ADM_s006
                                 + 1.53909 w_k1_AXP_s006
                                 + 1.52524 w_k1_DIS_s006
                                 + 1.47601 w_k1_FDX_s006
                                 + 1.22822 w_k1_IBM_s006
                                 + 1.29946 w_k1_INTC_s006
                                 + 1.25517 w_k1_KO_s006 + 1.54316 w_k1_MAT_s006
                                 + 1.4689 w_k1_MSFT_s006 + 1.10312 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.44428 w_k1_ADM_s007
                                 + 1.02963 w_k1_AXP_s007
                                 + 0.972213 w_k1_DIS_s007
                                 + 1.1208 w_k1_FDX_s007
                                 + 0.778988 w_k1_IBM_s007
                                 + 1.1553 w_k1_INTC_s007
                                 + 0.847087 w_k1_KO_s007
                                 + 0.59268 w_k1_MAT_s007
                                 + 1.19404 w_k1_MSFT_s007
                                 + 0.999508 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.53173 w_k1_ADM_s007
                                 + 1.32116 w_k1_AXP_s007
                                 + 1.08566 w_k1_DIS_s007
                                 + 1.17577 w_k1_FDX_s007
                                 + 1.09414 w_k1_IBM_s007
                                 + 1.08048 w_k1_INTC_s007
                                 + 1.03099 w_k1_KO_s007
                                 + 0.953404 w_k1_MAT_s007
                                 + 0.902043 w_k1_MSFT_s007
                                 + 1.04962 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.83794 w_k1_ADM_s007
                                 + 0.760336 w_k1_AXP_s007
                                 + 0.781114 w_k1_DIS_s007
                                 + 0.804392 w_k1_FDX_s007
                                 + 0.937699 w_k1_IBM_s007
                                 + 1.05932 w_k1_INTC_s007
                                 + 0.975741 w_k1_KO_s007
                                 + 0.669064 w_k1_MAT_s007
                                 + 0.963834 w_k1_MSFT_s007
                                 + 0.889854 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.05519 w_k1_ADM_s007
                                 + 0.681164 w_k1_AXP_s007
                                 + 0.65573 w_k1_DIS_s007
                                 + 0.795145 w_k1_FDX_s007
                                 + 0.658612 w_k1_IBM_s007
                                 + 0.917761 w_k1_INTC_s007
                                 + 0.73101 w_k1_KO_s007
                                 + 0.471571 w_k1_MAT_s007
                                 + 0.897137 w_k1_MSFT_s007
                                 + 0.73514 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.863916 w_k1_ADM_s007
                                 + 0.67905 w_k1_AXP_s007
                                 + 0.907393 w_k1_DIS_s007
                                 + 0.939748 w_k1_FDX_s007
                                 + 0.810947 w_k1_IBM_s007
                                 + 0.918518 w_k1_INTC_s007
                                 + 0.802349 w_k1_KO_s007
                                 + 0.693886 w_k1_MAT_s007
                                 + 1.01555 w_k1_MSFT_s007
                                 + 0.914368 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.05025 w_k1_ADM_s007
                                 + 1.08087 w_k1_AXP_s007
                                 + 1.35497 w_k1_DIS_s007
                                 + 1.08082 w_k1_FDX_s007
                                 + 1.70613 w_k1_IBM_s007
                                 + 1.17677 w_k1_INTC_s007
                                 + 1.31438 w_k1_KO_s007 + 1.29716 w_k1_MAT_s007
                                 + 1.10299 w_k1_MSFT_s007
                                 + 1.13241 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.20924 w_k1_ADM_s007
                                 + 1.10766 w_k1_AXP_s007
                                 + 1.67423 w_k1_DIS_s007
                                 + 1.37426 w_k1_FDX_s007 + 1.1565 w_k1_IBM_s007
                                 + 1.32619 w_k1_INTC_s007
                                 + 1.34118 w_k1_KO_s007
                                 + 0.871163 w_k1_MAT_s007
                                 + 1.26502 w_k1_MSFT_s007
                                 + 1.11604 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.81534 w_k1_ADM_s007
                                 + 0.873915 w_k1_AXP_s007 + 1.007 w_k1_DIS_s007
                                 + 0.958707 w_k1_FDX_s007
                                 + 1.24992 w_k1_IBM_s007
                                 + 1.00387 w_k1_INTC_s007
                                 + 1.05188 w_k1_KO_s007 + 1.15633 w_k1_MAT_s007
                                 + 1.03982 w_k1_MSFT_s007
                                 + 1.00421 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.920807 w_k1_ADM_s010
                                 + 0.686428 w_k1_AXP_s010
                                 + 0.817805 w_k1_DIS_s010
                                 + 0.914981 w_k1_FDX_s010
                                 + 0.729457 w_k1_IBM_s010
                                 + 0.845737 w_k1_INTC_s010
                                 + 0.836993 w_k1_KO_s010
                                 + 0.687311 w_k1_MAT_s010
                                 + 0.899045 w_k1_MSFT_s010
                                 + 0.848517 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.29204 w_k1_ADM_s010
                                 + 1.1416 w_k1_AXP_s010 + 1.07915 w_k1_DIS_s010
                                 + 1.19729 w_k1_FDX_s010
                                 + 0.926759 w_k1_IBM_s010
                                 + 0.831752 w_k1_INTC_s010
                                 + 0.957744 w_k1_KO_s010
                                 + 1.08238 w_k1_MAT_s010
                                 + 0.876731 w_k1_MSFT_s010
                                 + 0.951095 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.856027 w_k1_ADM_s010
                                 + 0.91045 w_k1_AXP_s010
                                 + 0.689045 w_k1_DIS_s010
                                 + 0.801529 w_k1_FDX_s010
                                 + 0.964789 w_k1_IBM_s010
                                 + 0.829467 w_k1_INTC_s010
                                 + 0.812607 w_k1_KO_s010
                                 + 1.07909 w_k1_MAT_s010
                                 + 0.69605 w_k1_MSFT_s010
                                 + 0.793438 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10166 w_k1_ADM_s010
                                 + 0.970971 w_k1_AXP_s010
                                 + 1.4287 w_k1_DIS_s010 + 1.12728 w_k1_FDX_s010
                                 + 0.983377 w_k1_IBM_s010
                                 + 0.723829 w_k1_INTC_s010
                                 + 1.08942 w_k1_KO_s010 + 1.33646 w_k1_MAT_s010
                                 + 0.967076 w_k1_MSFT_s010
                                 + 1.0161 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.811849 w_k1_ADM_s010
                                 + 0.786386 w_k1_AXP_s010
                                 + 0.647669 w_k1_DIS_s010
                                 + 0.694489 w_k1_FDX_s010
                                 + 1.12033 w_k1_IBM_s010
                                 + 0.869885 w_k1_INTC_s010
                                 + 0.892229 w_k1_KO_s010
                                 + 0.988755 w_k1_MAT_s010
                                 + 0.691715 w_k1_MSFT_s010
                                 + 0.753043 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.973958 w_k1_ADM_s010
                                 + 1.2039 w_k1_AXP_s010 + 1.0646 w_k1_DIS_s010
                                 + 0.985584 w_k1_FDX_s010
                                 + 1.33959 w_k1_IBM_s010
                                 + 0.96064 w_k1_INTC_s010
                                 + 1.12573 w_k1_KO_s010 + 1.68696 w_k1_MAT_s010
                                 + 1.01978 w_k1_MSFT_s010
                                 + 0.947228 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.626475 w_k1_ADM_s010
                                 + 0.570515 w_k1_AXP_s010
                                 + 0.587191 w_k1_DIS_s010
                                 + 0.663478 w_k1_FDX_s010
                                 + 0.506653 w_k1_IBM_s010
                                 + 0.600917 w_k1_INTC_s010
                                 + 0.659827 w_k1_KO_s010
                                 + 0.579948 w_k1_MAT_s010
                                 + 0.640637 w_k1_MSFT_s010
                                 + 0.586164 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.3416 w_k1_ADM_s010
                                 + 1.73315 w_k1_AXP_s010
                                 + 1.87902 w_k1_DIS_s010 + 1.384 w_k1_FDX_s010
                                 + 1.39475 w_k1_IBM_s010
                                 + 1.45677 w_k1_INTC_s010
                                 + 1.45591 w_k1_KO_s010 + 2.05308 w_k1_MAT_s010
                                 + 1.36932 w_k1_MSFT_s010
                                 + 1.22759 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.980278 w_k1_ADM_s011
                                 + 1.38753 w_k1_AXP_s011
                                 + 1.13678 w_k1_DIS_s011
                                 + 1.12151 w_k1_FDX_s011 + 1.1087 w_k1_IBM_s011
                                 + 1.10414 w_k1_INTC_s011
                                 + 1.05311 w_k1_KO_s011 + 1.49174 w_k1_MAT_s011
                                 + 0.916578 w_k1_MSFT_s011
                                 + 1.04303 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.835029 w_k1_ADM_s011
                                 + 0.846966 w_k1_AXP_s011
                                 + 0.807116 w_k1_DIS_s011
                                 + 0.870539 w_k1_FDX_s011
                                 + 0.966529 w_k1_IBM_s011
                                 + 0.741751 w_k1_INTC_s011
                                 + 0.868353 w_k1_KO_s011 + 1.4904 w_k1_MAT_s011
                                 + 0.786589 w_k1_MSFT_s011
                                 + 0.957859 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.995696 w_k1_ADM_s011
                                 + 0.837371 w_k1_AXP_s011
                                 + 0.896571 w_k1_DIS_s011
                                 + 0.965855 w_k1_FDX_s011
                                 + 0.772226 w_k1_IBM_s011
                                 + 0.841767 w_k1_INTC_s011
                                 + 0.871955 w_k1_KO_s011
                                 + 0.88038 w_k1_MAT_s011
                                 + 0.856157 w_k1_MSFT_s011
                                 + 0.882497 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.12373 w_k1_ADM_s011
                                 + 1.01038 w_k1_AXP_s011
                                 + 1.17345 w_k1_DIS_s011
                                 + 1.02172 w_k1_FDX_s011
                                 + 1.42825 w_k1_IBM_s011
                                 + 0.922115 w_k1_INTC_s011
                                 + 1.18952 w_k1_KO_s011 + 1.73233 w_k1_MAT_s011
                                 + 1.06628 w_k1_MSFT_s011
                                 + 1.11959 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.06754 w_k1_ADM_s011
                                 + 0.951308 w_k1_AXP_s011
                                 + 1.18984 w_k1_DIS_s011 + 1.052 w_k1_FDX_s011
                                 + 1.0387 w_k1_IBM_s011
                                 + 0.961002 w_k1_INTC_s011
                                 + 1.0542 w_k1_KO_s011 + 0.955551 w_k1_MAT_s011
                                 + 1.07172 w_k1_MSFT_s011
                                 + 1.01736 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.375 w_k1_ADM_s011
                                 + 0.86992 w_k1_AXP_s011
                                 + 1.23843 w_k1_DIS_s011
                                 + 1.20343 w_k1_FDX_s011
                                 + 0.777051 w_k1_IBM_s011
                                 + 0.938083 w_k1_INTC_s011
                                 + 1.17055 w_k1_KO_s011
                                 + 0.727537 w_k1_MAT_s011
                                 + 1.19435 w_k1_MSFT_s011
                                 + 1.08306 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.42935 w_k1_ADM_s011
                                 + 1.3783 w_k1_AXP_s011 + 1.81385 w_k1_DIS_s011
                                 + 1.32971 w_k1_FDX_s011
                                 + 1.18631 w_k1_IBM_s011
                                 + 1.20573 w_k1_INTC_s011
                                 + 1.48861 w_k1_KO_s011 + 1.6011 w_k1_MAT_s011
                                 + 1.40883 w_k1_MSFT_s011
                                 + 1.17568 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.13462 w_k1_ADM_s011
                                 + 1.74981 w_k1_AXP_s011
                                 + 1.30217 w_k1_DIS_s011
                                 + 1.36637 w_k1_FDX_s011
                                 + 1.36889 w_k1_IBM_s011
                                 + 1.06547 w_k1_INTC_s011 + 1.1721 w_k1_KO_s011
                                 + 2.20303 w_k1_MAT_s011
                                 + 1.12407 w_k1_MSFT_s011
                                 + 1.13286 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.28652 w_k1_ADM_s012
                                 + 1.16292 w_k1_AXP_s012
                                 + 1.03575 w_k1_DIS_s012
                                 + 1.11695 w_k1_FDX_s012
                                 + 0.923852 w_k1_IBM_s012
                                 + 1.01748 w_k1_INTC_s012
                                 + 0.993947 w_k1_KO_s012
                                 + 0.954493 w_k1_MAT_s012
                                 + 0.925084 w_k1_MSFT_s012
                                 + 1.04756 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.96827 w_k1_ADM_s012
                                 + 1.19273 w_k1_AXP_s012
                                 + 1.35886 w_k1_DIS_s012
                                 + 1.09529 w_k1_FDX_s012
                                 + 1.40215 w_k1_IBM_s012
                                 + 1.12916 w_k1_INTC_s012
                                 + 1.33596 w_k1_KO_s012 + 1.28142 w_k1_MAT_s012
                                 + 0.931757 w_k1_MSFT_s012
                                 + 1.15431 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.21112 w_k1_ADM_s012
                                 + 0.900362 w_k1_AXP_s012
                                 + 1.10646 w_k1_DIS_s012
                                 + 1.07464 w_k1_FDX_s012
                                 + 0.66063 w_k1_IBM_s012
                                 + 0.97207 w_k1_INTC_s012
                                 + 1.00748 w_k1_KO_s012
                                 + 0.674606 w_k1_MAT_s012
                                 + 1.03983 w_k1_MSFT_s012 + 1.0499 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.993388 w_k1_ADM_s012
                                 + 0.824303 w_k1_AXP_s012
                                 + 0.810935 w_k1_DIS_s012
                                 + 0.831031 w_k1_FDX_s012
                                 + 1.0672 w_k1_IBM_s012
                                 + 0.817655 w_k1_INTC_s012
                                 + 1.02311 w_k1_KO_s012 + 1.03875 w_k1_MAT_s012
                                 + 0.833234 w_k1_MSFT_s012
                                 + 0.982074 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.55729 w_k1_ADM_s012
                                 + 1.54582 w_k1_AXP_s012 + 1.5891 w_k1_DIS_s012
                                 + 1.37677 w_k1_FDX_s012
                                 + 1.18031 w_k1_IBM_s012
                                 + 1.14703 w_k1_INTC_s012
                                 + 1.28467 w_k1_KO_s012 + 1.35582 w_k1_MAT_s012
                                 + 1.15256 w_k1_MSFT_s012
                                 + 1.16496 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.936373 w_k1_ADM_s012
                                 + 0.721394 w_k1_AXP_s012
                                 + 0.738244 w_k1_DIS_s012
                                 + 0.825124 w_k1_FDX_s012
                                 + 0.740264 w_k1_IBM_s012
                                 + 0.751384 w_k1_INTC_s012
                                 + 0.889653 w_k1_KO_s012
                                 + 0.831057 w_k1_MAT_s012
                                 + 0.813681 w_k1_MSFT_s012
                                 + 0.828565 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.02026 w_k1_ADM_s012
                                 + 0.961341 w_k1_AXP_s012
                                 + 1.03683 w_k1_DIS_s012
                                 + 0.985344 w_k1_FDX_s012
                                 + 0.940027 w_k1_IBM_s012
                                 + 0.881544 w_k1_INTC_s012
                                 + 1.00897 w_k1_KO_s012
                                 + 0.951837 w_k1_MAT_s012
                                 + 0.96506 w_k1_MSFT_s012
                                 + 0.876579 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.987065 w_k1_ADM_s012
                                 + 1.27551 w_k1_AXP_s012
                                 + 1.01433 w_k1_DIS_s012
                                 + 1.04031 w_k1_FDX_s012
                                 + 1.45729 w_k1_IBM_s012
                                 + 0.794165 w_k1_INTC_s012
                                 + 1.13803 w_k1_KO_s012 + 1.55525 w_k1_MAT_s012
                                 + 0.881046 w_k1_MSFT_s012
                                 + 1.04799 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.22174 w_k1_ADM_s013
                                 + 1.04941 w_k1_AXP_s013
                                 + 0.697264 w_k1_DIS_s013
                                 + 0.867735 w_k1_FDX_s013
                                 + 1.06677 w_k1_IBM_s013
                                 + 0.887477 w_k1_INTC_s013
                                 + 0.822576 w_k1_KO_s013
                                 + 1.00001 w_k1_MAT_s013
                                 + 0.856334 w_k1_MSFT_s013
                                 + 0.943488 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34168 w_k1_ADM_s013
                                 + 1.64474 w_k1_AXP_s013
                                 + 1.43247 w_k1_DIS_s013 + 1.1786 w_k1_FDX_s013
                                 + 1.52408 w_k1_IBM_s013
                                 + 1.46882 w_k1_INTC_s013
                                 + 1.20939 w_k1_KO_s013 + 1.24496 w_k1_MAT_s013
                                 + 1.30789 w_k1_MSFT_s013
                                 + 1.22621 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.13734 w_k1_ADM_s013
                                 + 0.983301 w_k1_AXP_s013
                                 + 1.29536 w_k1_DIS_s013
                                 + 1.11494 w_k1_FDX_s013
                                 + 1.01217 w_k1_IBM_s013
                                 + 1.06507 w_k1_INTC_s013 + 1.0896 w_k1_KO_s013
                                 + 1.20064 w_k1_MAT_s013
                                 + 1.15992 w_k1_MSFT_s013
                                 + 1.11534 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.38283 w_k1_ADM_s013
                                 + 1.58216 w_k1_AXP_s013
                                 + 1.73674 w_k1_DIS_s013
                                 + 1.53522 w_k1_FDX_s013
                                 + 1.10304 w_k1_IBM_s013
                                 + 1.57524 w_k1_INTC_s013
                                 + 1.26359 w_k1_KO_s013 + 1.39512 w_k1_MAT_s013
                                 + 1.61156 w_k1_MSFT_s013
                                 + 1.31979 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.859509 w_k1_ADM_s013
                                 + 0.665084 w_k1_AXP_s013
                                 + 0.650292 w_k1_DIS_s013
                                 + 0.635243 w_k1_FDX_s013
                                 + 1.03626 w_k1_IBM_s013
                                 + 0.813449 w_k1_INTC_s013
                                 + 0.78897 w_k1_KO_s013
                                 + 0.940011 w_k1_MAT_s013
                                 + 0.767847 w_k1_MSFT_s013
                                 + 0.752589 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.95231 w_k1_ADM_s013
                                 + 0.866741 w_k1_AXP_s013
                                 + 0.843615 w_k1_DIS_s013
                                 + 0.958439 w_k1_FDX_s013
                                 + 0.713606 w_k1_IBM_s013
                                 + 0.889186 w_k1_INTC_s013
                                 + 0.71135 w_k1_KO_s013 + 0.69506 w_k1_MAT_s013
                                 + 1.02284 w_k1_MSFT_s013
                                 + 0.872287 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.02104 w_k1_ADM_s013
                                 + 1.01741 w_k1_AXP_s013
                                 + 0.960794 w_k1_DIS_s013
                                 + 0.93274 w_k1_FDX_s013
                                 + 1.21029 w_k1_IBM_s013
                                 + 1.11322 w_k1_INTC_s013
                                 + 1.07279 w_k1_KO_s013
                                 + 0.878615 w_k1_MAT_s013
                                 + 1.02759 w_k1_MSFT_s013
                                 + 0.991665 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.29008 w_k1_ADM_s013
                                 + 1.48552 w_k1_AXP_s013
                                 + 1.04699 w_k1_DIS_s013
                                 + 1.20452 w_k1_FDX_s013
                                 + 1.18167 w_k1_IBM_s013
                                 + 1.18262 w_k1_INTC_s013
                                 + 0.988463 w_k1_KO_s013
                                 + 1.42499 w_k1_MAT_s013
                                 + 1.11511 w_k1_MSFT_s013
                                 + 1.00918 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.18945 w_k1_ADM_s014
                                 + 1.46614 w_k1_AXP_s014
                                 + 1.27211 w_k1_DIS_s014
                                 + 1.23756 w_k1_FDX_s014
                                 + 1.17834 w_k1_IBM_s014
                                 + 1.43965 w_k1_INTC_s014
                                 + 1.12435 w_k1_KO_s014 + 1.33817 w_k1_MAT_s014
                                 + 1.12731 w_k1_MSFT_s014
                                 + 1.16682 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.844053 w_k1_ADM_s014
                                 + 0.853962 w_k1_AXP_s014
                                 + 0.680926 w_k1_DIS_s014
                                 + 0.847988 w_k1_FDX_s014
                                 + 0.959278 w_k1_IBM_s014
                                 + 0.889258 w_k1_INTC_s014
                                 + 0.840175 w_k1_KO_s014
                                 + 1.02315 w_k1_MAT_s014
                                 + 0.787192 w_k1_MSFT_s014
                                 + 0.99246 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.31899 w_k1_ADM_s014
                                 + 1.15475 w_k1_AXP_s014
                                 + 1.37641 w_k1_DIS_s014
                                 + 1.28419 w_k1_FDX_s014
                                 + 0.979753 w_k1_IBM_s014
                                 + 1.13725 w_k1_INTC_s014
                                 + 1.25588 w_k1_KO_s014
                                 + 0.921079 w_k1_MAT_s014
                                 + 1.13871 w_k1_MSFT_s014
                                 + 1.05755 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.938621 w_k1_ADM_s014
                                 + 1.06484 w_k1_AXP_s014
                                 + 0.84875 w_k1_DIS_s014
                                 + 0.85571 w_k1_FDX_s014
                                 + 1.42804 w_k1_IBM_s014
                                 + 1.03394 w_k1_INTC_s014
                                 + 1.06417 w_k1_KO_s014 + 1.23954 w_k1_MAT_s014
                                 + 0.865929 w_k1_MSFT_s014
                                 + 1.11177 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.13537 w_k1_ADM_s014
                                 + 1.05441 w_k1_AXP_s014
                                 + 0.896835 w_k1_DIS_s014
                                 + 1.06057 w_k1_FDX_s014
                                 + 0.834496 w_k1_IBM_s014
                                 + 1.14306 w_k1_INTC_s014
                                 + 0.877632 w_k1_KO_s014
                                 + 0.817552 w_k1_MAT_s014
                                 + 0.913297 w_k1_MSFT_s014
                                 + 1.08799 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.22131 w_k1_ADM_s014
                                 + 1.42899 w_k1_AXP_s014
                                 + 1.40687 w_k1_DIS_s014
                                 + 1.14152 w_k1_FDX_s014
                                 + 1.58355 w_k1_IBM_s014
                                 + 1.06816 w_k1_INTC_s014
                                 + 1.44946 w_k1_KO_s014 + 1.63653 w_k1_MAT_s014
                                 + 1.10748 w_k1_MSFT_s014
                                 + 1.27388 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.895209 w_k1_ADM_s014
                                 + 0.730326 w_k1_AXP_s014
                                 + 0.644479 w_k1_DIS_s014
                                 + 0.854784 w_k1_FDX_s014
                                 + 0.679542 w_k1_IBM_s014
                                 + 0.853146 w_k1_INTC_s014
                                 + 0.835519 w_k1_KO_s014
                                 + 0.76241 w_k1_MAT_s014
                                 + 0.821673 w_k1_MSFT_s014
                                 + 0.818548 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.641031 w_k1_ADM_s014
                                 + 0.848732 w_k1_AXP_s014
                                 + 0.921021 w_k1_DIS_s014
                                 + 0.91081 w_k1_FDX_s014
                                 + 1.19341 w_k1_IBM_s014
                                 + 0.918248 w_k1_INTC_s014
                                 + 0.991478 w_k1_KO_s014
                                 + 1.01895 w_k1_MAT_s014
                                 + 0.830155 w_k1_MSFT_s014
                                 + 1.05256 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.11717 w_k1_ADM_s015
                                 + 1.65953 w_k1_AXP_s015
                                 + 1.69732 w_k1_DIS_s015
                                 + 1.47919 w_k1_FDX_s015 + 1.0798 w_k1_IBM_s015
                                 + 1.20656 w_k1_INTC_s015
                                 + 1.16744 w_k1_KO_s015 + 1.10467 w_k1_MAT_s015
                                 + 1.31232 w_k1_MSFT_s015
                                 + 1.09108 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.073 w_k1_ADM_s015
                                 + 0.973059 w_k1_AXP_s015
                                 + 1.21697 w_k1_DIS_s015
                                 + 1.10721 w_k1_FDX_s015
                                 + 0.810652 w_k1_IBM_s015
                                 + 0.964862 w_k1_INTC_s015
                                 + 0.967236 w_k1_KO_s015
                                 + 0.972131 w_k1_MAT_s015
                                 + 0.954988 w_k1_MSFT_s015
                                 + 0.937479 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.27129 w_k1_ADM_s015
                                 + 1.27378 w_k1_AXP_s015
                                 + 0.940554 w_k1_DIS_s015
                                 + 1.10031 w_k1_FDX_s015
                                 + 0.769856 w_k1_IBM_s015
                                 + 0.910713 w_k1_INTC_s015
                                 + 0.78471 w_k1_KO_s015 + 1.00707 w_k1_MAT_s015
                                 + 0.976637 w_k1_MSFT_s015
                                 + 1.02077 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.947161 w_k1_ADM_s015
                                 + 0.683752 w_k1_AXP_s015
                                 + 0.882225 w_k1_DIS_s015
                                 + 0.839384 w_k1_FDX_s015
                                 + 1.03578 w_k1_IBM_s015
                                 + 0.770117 w_k1_INTC_s015
                                 + 0.993328 w_k1_KO_s015
                                 + 0.84577 w_k1_MAT_s015
                                 + 0.831507 w_k1_MSFT_s015
                                 + 0.960607 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.807629 w_k1_ADM_s015
                                 + 0.789754 w_k1_AXP_s015
                                 + 0.676988 w_k1_DIS_s015
                                 + 0.861398 w_k1_FDX_s015
                                 + 1.23747 w_k1_IBM_s015
                                 + 0.874299 w_k1_INTC_s015
                                 + 0.998077 w_k1_KO_s015
                                 + 1.46174 w_k1_MAT_s015
                                 + 0.886906 w_k1_MSFT_s015
                                 + 0.899661 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.89378 w_k1_ADM_s015
                                 + 1.11591 w_k1_AXP_s015
                                 + 1.31077 w_k1_DIS_s015
                                 + 1.14918 w_k1_FDX_s015
                                 + 1.34926 w_k1_IBM_s015
                                 + 1.0873 w_k1_INTC_s015 + 1.04788 w_k1_KO_s015
                                 + 1.17652 w_k1_MAT_s015
                                 + 1.05841 w_k1_MSFT_s015
                                 + 0.975463 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.611914 w_k1_ADM_s015
                                 + 0.526318 w_k1_AXP_s015
                                 + 0.615736 w_k1_DIS_s015
                                 + 0.710594 w_k1_FDX_s015
                                 + 0.728509 w_k1_IBM_s015
                                 + 0.660493 w_k1_INTC_s015
                                 + 0.674291 w_k1_KO_s015
                                 + 0.785953 w_k1_MAT_s015
                                 + 0.62677 w_k1_MSFT_s015
                                 + 0.741531 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.09138 w_k1_ADM_s015
                                 + 1.32469 w_k1_AXP_s015
                                 + 1.81184 w_k1_DIS_s015
                                 + 1.33653 w_k1_FDX_s015
                                 + 1.89726 w_k1_IBM_s015
                                 + 1.1552 w_k1_INTC_s015 + 1.51565 w_k1_KO_s015
                                 + 1.82174 w_k1_MAT_s015
                                 + 1.17377 w_k1_MSFT_s015
                                 + 1.24276 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.83614 w_k1_ADM_s016
                                 + 1.08654 w_k1_AXP_s016
                                 + 1.60326 w_k1_DIS_s016
                                 + 1.26008 w_k1_FDX_s016
                                 + 1.20111 w_k1_IBM_s016
                                 + 1.48257 w_k1_INTC_s016
                                 + 1.31735 w_k1_KO_s016 + 1.58295 w_k1_MAT_s016
                                 + 1.41876 w_k1_MSFT_s016
                                 + 1.16727 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.07841 w_k1_ADM_s016
                                 + 0.942512 w_k1_AXP_s016
                                 + 1.04376 w_k1_DIS_s016
                                 + 0.974691 w_k1_FDX_s016
                                 + 1.62931 w_k1_IBM_s016
                                 + 0.925423 w_k1_INTC_s016
                                 + 1.18425 w_k1_KO_s016 + 1.61155 w_k1_MAT_s016
                                 + 1.04195 w_k1_MSFT_s016
                                 + 1.08137 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.02268 w_k1_ADM_s016
                                 + 1.10294 w_k1_AXP_s016
                                 + 1.04611 w_k1_DIS_s016
                                 + 1.02193 w_k1_FDX_s016
                                 + 1.23129 w_k1_IBM_s016
                                 + 0.843616 w_k1_INTC_s016
                                 + 1.17938 w_k1_KO_s016 + 1.43891 w_k1_MAT_s016
                                 + 0.994979 w_k1_MSFT_s016
                                 + 1.04496 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.883548 w_k1_ADM_s016
                                 + 0.802464 w_k1_AXP_s016
                                 + 0.899178 w_k1_DIS_s016
                                 + 0.949986 w_k1_FDX_s016
                                 + 0.838901 w_k1_IBM_s016
                                 + 0.923836 w_k1_INTC_s016
                                 + 0.899883 w_k1_KO_s016
                                 + 1.05076 w_k1_MAT_s016
                                 + 1.00814 w_k1_MSFT_s016
                                 + 0.980728 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.08256 w_k1_ADM_s016
                                 + 0.881414 w_k1_AXP_s016
                                 + 1.28626 w_k1_DIS_s016 + 1.0166 w_k1_FDX_s016
                                 + 1.02395 w_k1_IBM_s016
                                 + 0.959497 w_k1_INTC_s016
                                 + 1.11338 w_k1_KO_s016 + 1.21377 w_k1_MAT_s016
                                 + 0.987107 w_k1_MSFT_s016
                                 + 1.02746 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.667921 w_k1_ADM_s016
                                 + 0.603498 w_k1_AXP_s016
                                 + 0.543218 w_k1_DIS_s016
                                 + 0.696476 w_k1_FDX_s016
                                 + 1.0559 w_k1_IBM_s016
                                 + 0.73995 w_k1_INTC_s016
                                 + 0.762817 w_k1_KO_s016
                                 + 0.898722 w_k1_MAT_s016
                                 + 0.721139 w_k1_MSFT_s016
                                 + 0.759296 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.902145 w_k1_ADM_s016
                                 + 0.771867 w_k1_AXP_s016
                                 + 0.771786 w_k1_DIS_s016
                                 + 0.882562 w_k1_FDX_s016
                                 + 0.609043 w_k1_IBM_s016
                                 + 0.852769 w_k1_INTC_s016
                                 + 0.778103 w_k1_KO_s016
                                 + 0.56996 w_k1_MAT_s016
                                 + 0.875446 w_k1_MSFT_s016
                                 + 0.814797 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.3769 w_k1_ADM_s016
                                 + 1.03711 w_k1_AXP_s016 + 1.4144 w_k1_DIS_s016
                                 + 1.35314 w_k1_FDX_s016
                                 + 0.839128 w_k1_IBM_s016
                                 + 1.11372 w_k1_INTC_s016
                                 + 1.16221 w_k1_KO_s016 + 1.15923 w_k1_MAT_s016
                                 + 1.28606 w_k1_MSFT_s016
                                 + 1.16216 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.939445 w_k1_ADM_s017
                                 + 1.29113 w_k1_AXP_s017
                                 + 1.48286 w_k1_DIS_s017
                                 + 1.06454 w_k1_FDX_s017
                                 + 1.55268 w_k1_IBM_s017
                                 + 1.18731 w_k1_INTC_s017
                                 + 1.40684 w_k1_KO_s017 + 1.63598 w_k1_MAT_s017
                                 + 1.22033 w_k1_MSFT_s017 + 1.2036 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12776 w_k1_ADM_s017
                                 + 1.7076 w_k1_AXP_s017 + 1.7136 w_k1_DIS_s017
                                 + 1.35122 w_k1_FDX_s017
                                 + 1.64761 w_k1_IBM_s017
                                 + 1.11705 w_k1_INTC_s017
                                 + 1.55337 w_k1_KO_s017 + 1.77269 w_k1_MAT_s017
                                 + 1.27615 w_k1_MSFT_s017
                                 + 1.17513 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.79862 w_k1_ADM_s017
                                 + 1.10035 w_k1_AXP_s017
                                 + 0.920564 w_k1_DIS_s017
                                 + 0.945438 w_k1_FDX_s017
                                 + 1.30253 w_k1_IBM_s017
                                 + 0.939273 w_k1_INTC_s017
                                 + 1.2305 w_k1_KO_s017 + 1.48474 w_k1_MAT_s017
                                 + 0.954139 w_k1_MSFT_s017
                                 + 0.985046 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.866644 w_k1_ADM_s017
                                 + 0.871086 w_k1_AXP_s017
                                 + 0.905352 w_k1_DIS_s017
                                 + 0.87717 w_k1_FDX_s017
                                 + 1.02749 w_k1_IBM_s017
                                 + 0.939507 w_k1_INTC_s017
                                 + 1.02882 w_k1_KO_s017
                                 + 0.951933 w_k1_MAT_s017
                                 + 0.975138 w_k1_MSFT_s017
                                 + 0.875489 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.873633 w_k1_ADM_s017
                                 + 1.11693 w_k1_AXP_s017
                                 + 1.57134 w_k1_DIS_s017
                                 + 1.22784 w_k1_FDX_s017
                                 + 0.940817 w_k1_IBM_s017
                                 + 1.06252 w_k1_INTC_s017
                                 + 1.16737 w_k1_KO_s017
                                 + 0.716366 w_k1_MAT_s017
                                 + 1.13345 w_k1_MSFT_s017 + 1.0787 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.854093 w_k1_ADM_s017
                                 + 1.13948 w_k1_AXP_s017
                                 + 1.03571 w_k1_DIS_s017
                                 + 1.04426 w_k1_FDX_s017
                                 + 0.937906 w_k1_IBM_s017
                                 + 0.960523 w_k1_INTC_s017
                                 + 0.992588 w_k1_KO_s017
                                 + 1.08533 w_k1_MAT_s017
                                 + 1.02502 w_k1_MSFT_s017
                                 + 0.944401 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.672164 w_k1_ADM_s017
                                 + 0.654687 w_k1_AXP_s017
                                 + 0.685222 w_k1_DIS_s017
                                 + 0.777848 w_k1_FDX_s017
                                 + 0.727319 w_k1_IBM_s017
                                 + 0.82568 w_k1_INTC_s017
                                 + 0.829793 w_k1_KO_s017
                                 + 0.555861 w_k1_MAT_s017
                                 + 0.775947 w_k1_MSFT_s017
                                 + 0.772051 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.864772 w_k1_ADM_s017
                                 + 0.910547 w_k1_AXP_s017
                                 + 0.767823 w_k1_DIS_s017
                                 + 0.913577 w_k1_FDX_s017
                                 + 0.67444 w_k1_IBM_s017
                                 + 0.716002 w_k1_INTC_s017
                                 + 0.673785 w_k1_KO_s017
                                 + 0.765373 w_k1_MAT_s017
                                 + 0.753738 w_k1_MSFT_s017
                                 + 0.83752 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.9283 w_k1_ADM_s100
                                 + 0.900571 w_k1_AXP_s100
                                 + 1.13226 w_k1_DIS_s100
                                 + 0.98172 w_k1_FDX_s100
                                 + 0.78367 w_k1_IBM_s100
                                 + 1.04976 w_k1_INTC_s100
                                 + 0.916483 w_k1_KO_s100
                                 + 0.691885 w_k1_MAT_s100
                                 + 1.04471 w_k1_MSFT_s100
                                 + 0.886222 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.830047 w_k1_ADM_s100
                                 + 0.745377 w_k1_AXP_s100
                                 + 0.997685 w_k1_DIS_s100
                                 + 0.915502 w_k1_FDX_s100
                                 + 0.914795 w_k1_IBM_s100
                                 + 1.0425 w_k1_INTC_s100
                                 + 0.967381 w_k1_KO_s100
                                 + 0.718198 w_k1_MAT_s100
                                 + 1.04958 w_k1_MSFT_s100
                                 + 0.969544 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.10576 w_k1_ADM_s100
                                 + 1.61329 w_k1_AXP_s100
                                 + 1.14431 w_k1_DIS_s100
                                 + 1.23036 w_k1_FDX_s100
                                 + 1.31971 w_k1_IBM_s100
                                 + 1.20616 w_k1_INTC_s100 + 1.1632 w_k1_KO_s100
                                 + 1.81147 w_k1_MAT_s100
                                 + 1.08665 w_k1_MSFT_s100
                                 + 1.14978 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.840368 w_k1_ADM_s100
                                 + 0.735961 w_k1_AXP_s100
                                 + 0.659649 w_k1_DIS_s100
                                 + 0.786712 w_k1_FDX_s100
                                 + 0.849302 w_k1_IBM_s100
                                 + 0.94277 w_k1_INTC_s100
                                 + 0.755219 w_k1_KO_s100
                                 + 0.612499 w_k1_MAT_s100
                                 + 0.841798 w_k1_MSFT_s100
                                 + 0.845314 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.01471 w_k1_ADM_s100
                                 + 0.923317 w_k1_AXP_s100
                                 + 0.840576 w_k1_DIS_s100
                                 + 0.94829 w_k1_FDX_s100
                                 + 1.07073 w_k1_IBM_s100
                                 + 0.91728 w_k1_INTC_s100
                                 + 0.843773 w_k1_KO_s100
                                 + 0.86829 w_k1_MAT_s100
                                 + 0.92715 w_k1_MSFT_s100
                                 + 0.981087 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.781081 w_k1_ADM_s100
                                 + 1.08907 w_k1_AXP_s100
                                 + 1.07447 w_k1_DIS_s100
                                 + 0.956405 w_k1_FDX_s100
                                 + 1.34155 w_k1_IBM_s100
                                 + 1.2225 w_k1_INTC_s100 + 1.08067 w_k1_KO_s100
                                 + 1.19535 w_k1_MAT_s100
                                 + 0.968281 w_k1_MSFT_s100
                                 + 1.12121 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16185 w_k1_ADM_s100
                                 + 1.15264 w_k1_AXP_s100
                                 + 1.69815 w_k1_DIS_s100
                                 + 1.36154 w_k1_FDX_s100
                                 + 1.21541 w_k1_IBM_s100
                                 + 1.30399 w_k1_INTC_s100
                                 + 1.28363 w_k1_KO_s100 + 1.0764 w_k1_MAT_s100
                                 + 1.43919 w_k1_MSFT_s100
                                 + 1.13215 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.08615 w_k1_ADM_s100
                                 + 1.19335 w_k1_AXP_s100
                                 + 1.10637 w_k1_DIS_s100
                                 + 1.14742 w_k1_FDX_s100
                                 + 1.04559 w_k1_IBM_s100
                                 + 1.07738 w_k1_INTC_s100
                                 + 0.925051 w_k1_KO_s100 + 1.3217 w_k1_MAT_s100
                                 + 1.07524 w_k1_MSFT_s100
                                 + 1.02301 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.34165 w_k1_ADM_s101
                                 + 1.07167 w_k1_AXP_s101
                                 + 1.16578 w_k1_DIS_s101
                                 + 1.07022 w_k1_FDX_s101
                                 + 1.17798 w_k1_IBM_s101
                                 + 0.885812 w_k1_INTC_s101
                                 + 1.06133 w_k1_KO_s101 + 1.1846 w_k1_MAT_s101
                                 + 0.922162 w_k1_MSFT_s101
                                 + 1.03219 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.992041 w_k1_ADM_s101
                                 + 0.76294 w_k1_AXP_s101
                                 + 0.960066 w_k1_DIS_s101
                                 + 0.928021 w_k1_FDX_s101
                                 + 0.648185 w_k1_IBM_s101
                                 + 0.872229 w_k1_INTC_s101
                                 + 0.793722 w_k1_KO_s101
                                 + 0.740934 w_k1_MAT_s101
                                 + 0.988706 w_k1_MSFT_s101 + 1.023 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.878177 w_k1_ADM_s101
                                 + 0.607449 w_k1_AXP_s101
                                 + 0.800101 w_k1_DIS_s101
                                 + 0.754739 w_k1_FDX_s101
                                 + 0.974316 w_k1_IBM_s101
                                 + 0.611038 w_k1_INTC_s101
                                 + 0.896702 w_k1_KO_s101
                                 + 0.864908 w_k1_MAT_s101
                                 + 0.768704 w_k1_MSFT_s101
                                 + 0.805855 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.22411 w_k1_ADM_s101
                                 + 1.34087 w_k1_AXP_s101
                                 + 1.37399 w_k1_DIS_s101 + 1.2789 w_k1_FDX_s101
                                 + 1.59005 w_k1_IBM_s101
                                 + 0.986033 w_k1_INTC_s101
                                 + 1.37584 w_k1_KO_s101 + 1.80232 w_k1_MAT_s101
                                 + 1.06758 w_k1_MSFT_s101
                                 + 1.18629 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.38886 w_k1_ADM_s101
                                 + 0.828655 w_k1_AXP_s101
                                 + 1.22339 w_k1_DIS_s101
                                 + 1.18547 w_k1_FDX_s101
                                 + 0.732982 w_k1_IBM_s101
                                 + 0.882431 w_k1_INTC_s101
                                 + 0.903064 w_k1_KO_s101
                                 + 0.698765 w_k1_MAT_s101
                                 + 1.04352 w_k1_MSFT_s101
                                 + 1.02509 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.859359 w_k1_ADM_s101
                                 + 0.586175 w_k1_AXP_s101
                                 + 0.654387 w_k1_DIS_s101
                                 + 0.702622 w_k1_FDX_s101
                                 + 0.690173 w_k1_IBM_s101
                                 + 0.58499 w_k1_INTC_s101
                                 + 0.80136 w_k1_KO_s101
                                 + 0.641934 w_k1_MAT_s101
                                 + 0.750579 w_k1_MSFT_s101
                                 + 0.724334 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.06856 w_k1_ADM_s101
                                 + 1.26621 w_k1_AXP_s101
                                 + 1.17373 w_k1_DIS_s101 + 1.2047 w_k1_FDX_s101
                                 + 0.983863 w_k1_IBM_s101
                                 + 1.08744 w_k1_INTC_s101
                                 + 1.12978 w_k1_KO_s101 + 1.14148 w_k1_MAT_s101
                                 + 1.28926 w_k1_MSFT_s101
                                 + 1.16787 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.832024 w_k1_ADM_s101
                                 + 0.705041 w_k1_AXP_s101
                                 + 0.754947 w_k1_DIS_s101
                                 + 0.886632 w_k1_FDX_s101
                                 + 0.856553 w_k1_IBM_s101
                                 + 0.894534 w_k1_INTC_s101
                                 + 0.820839 w_k1_KO_s101
                                 + 0.940699 w_k1_MAT_s101
                                 + 0.805127 w_k1_MSFT_s101
                                 + 0.93566 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.88094 w_k1_ADM_s102
                                 + 1.11251 w_k1_AXP_s102
                                 + 0.892164 w_k1_DIS_s102
                                 + 1.01295 w_k1_FDX_s102
                                 + 0.846835 w_k1_IBM_s102
                                 + 1.03643 w_k1_INTC_s102
                                 + 0.92416 w_k1_KO_s102
                                 + 0.938932 w_k1_MAT_s102
                                 + 1.00763 w_k1_MSFT_s102
                                 + 0.952275 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.135 w_k1_ADM_s102
                                 + 1.40101 w_k1_AXP_s102 + 1.527 w_k1_DIS_s102
                                 + 1.16178 w_k1_FDX_s102
                                 + 1.76322 w_k1_IBM_s102
                                 + 1.52804 w_k1_INTC_s102 + 1.7129 w_k1_KO_s102
                                 + 1.55069 w_k1_MAT_s102
                                 + 1.38403 w_k1_MSFT_s102
                                 + 1.27347 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.96504 w_k1_ADM_s102
                                 + 1.1939 w_k1_AXP_s102 + 1.15919 w_k1_DIS_s102
                                 + 1.07868 w_k1_FDX_s102
                                 + 1.24117 w_k1_IBM_s102
                                 + 1.14778 w_k1_INTC_s102
                                 + 1.18356 w_k1_KO_s102 + 1.30924 w_k1_MAT_s102
                                 + 0.989552 w_k1_MSFT_s102
                                 + 1.12249 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.52475 w_k1_ADM_s102
                                 + 1.59605 w_k1_AXP_s102
                                 + 1.40626 w_k1_DIS_s102
                                 + 1.35279 w_k1_FDX_s102
                                 + 1.27918 w_k1_IBM_s102
                                 + 1.37366 w_k1_INTC_s102
                                 + 1.10756 w_k1_KO_s102 + 1.0701 w_k1_MAT_s102
                                 + 1.19492 w_k1_MSFT_s102
                                 + 1.04651 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.764444 w_k1_ADM_s102
                                 + 0.799621 w_k1_AXP_s102
                                 + 0.673319 w_k1_DIS_s102
                                 + 0.716532 w_k1_FDX_s102
                                 + 0.932434 w_k1_IBM_s102
                                 + 1.0305 w_k1_INTC_s102
                                 + 0.863671 w_k1_KO_s102
                                 + 0.860518 w_k1_MAT_s102
                                 + 0.764847 w_k1_MSFT_s102
                                 + 0.822993 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.07272 w_k1_ADM_s102
                                 + 0.743823 w_k1_AXP_s102
                                 + 1.13878 w_k1_DIS_s102 + 1.0371 w_k1_FDX_s102
                                 + 0.743242 w_k1_IBM_s102
                                 + 1.13856 w_k1_INTC_s102
                                 + 0.960962 w_k1_KO_s102
                                 + 0.588631 w_k1_MAT_s102
                                 + 1.07077 w_k1_MSFT_s102
                                 + 0.885305 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.34124 w_k1_ADM_s102
                                 + 1.12082 w_k1_AXP_s102
                                 + 1.76212 w_k1_DIS_s102
                                 + 1.32444 w_k1_FDX_s102
                                 + 0.849189 w_k1_IBM_s102
                                 + 1.45082 w_k1_INTC_s102
                                 + 1.32815 w_k1_KO_s102 + 0.81183 w_k1_MAT_s102
                                 + 1.45075 w_k1_MSFT_s102
                                 + 1.19147 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.14638 w_k1_ADM_s102
                                 + 1.02315 w_k1_AXP_s102
                                 + 0.968682 w_k1_DIS_s102
                                 + 0.920709 w_k1_FDX_s102
                                 + 1.25669 w_k1_IBM_s102
                                 + 1.2062 w_k1_INTC_s102 + 1.12766 w_k1_KO_s102
                                 + 1.05099 w_k1_MAT_s102
                                 + 1.08049 w_k1_MSFT_s102
                                 + 0.886584 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.0803 w_k1_ADM_s103
                                 + 1.10857 w_k1_AXP_s103
                                 + 1.19588 w_k1_DIS_s103
                                 + 1.09357 w_k1_FDX_s103
                                 + 1.44151 w_k1_IBM_s103
                                 + 1.14488 w_k1_INTC_s103
                                 + 1.20436 w_k1_KO_s103 + 1.36617 w_k1_MAT_s103
                                 + 1.01726 w_k1_MSFT_s103
                                 + 0.965867 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.824754 w_k1_ADM_s103
                                 + 0.95637 w_k1_AXP_s103
                                 + 0.751196 w_k1_DIS_s103
                                 + 0.783559 w_k1_FDX_s103
                                 + 1.11289 w_k1_IBM_s103
                                 + 1.00075 w_k1_INTC_s103
                                 + 0.836195 w_k1_KO_s103
                                 + 1.22278 w_k1_MAT_s103
                                 + 0.891859 w_k1_MSFT_s103
                                 + 0.932238 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.9756 w_k1_ADM_s103
                                 + 0.69915 w_k1_AXP_s103
                                 + 1.10604 w_k1_DIS_s103
                                 + 0.812962 w_k1_FDX_s103
                                 + 1.39728 w_k1_IBM_s103
                                 + 1.00051 w_k1_INTC_s103
                                 + 1.22705 w_k1_KO_s103 + 1.17056 w_k1_MAT_s103
                                 + 1.00131 w_k1_MSFT_s103
                                 + 0.974071 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.23016 w_k1_ADM_s103
                                 + 1.39188 w_k1_AXP_s103
                                 + 1.67089 w_k1_DIS_s103
                                 + 1.24422 w_k1_FDX_s103
                                 + 1.70312 w_k1_IBM_s103
                                 + 1.30662 w_k1_INTC_s103
                                 + 1.46387 w_k1_KO_s103 + 1.63209 w_k1_MAT_s103
                                 + 1.12623 w_k1_MSFT_s103
                                 + 1.16954 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.32051 w_k1_ADM_s103
                                 + 1.10295 w_k1_AXP_s103
                                 + 1.15596 w_k1_DIS_s103
                                 + 1.25416 w_k1_FDX_s103
                                 + 0.708531 w_k1_IBM_s103
                                 + 1.17704 w_k1_INTC_s103
                                 + 0.884398 w_k1_KO_s103
                                 + 0.843596 w_k1_MAT_s103
                                 + 1.2164 w_k1_MSFT_s103 + 1.07932 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.923111 w_k1_ADM_s103
                                 + 0.663634 w_k1_AXP_s103
                                 + 0.917032 w_k1_DIS_s103
                                 + 0.909674 w_k1_FDX_s103
                                 + 0.704612 w_k1_IBM_s103
                                 + 0.842987 w_k1_INTC_s103
                                 + 0.642468 w_k1_KO_s103
                                 + 0.668587 w_k1_MAT_s103
                                 + 0.867597 w_k1_MSFT_s103
                                 + 0.799016 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.708126 w_k1_ADM_s103
                                 + 0.561486 w_k1_AXP_s103
                                 + 0.701368 w_k1_DIS_s103
                                 + 0.750624 w_k1_FDX_s103
                                 + 0.845455 w_k1_IBM_s103
                                 + 0.736742 w_k1_INTC_s103
                                 + 0.71367 w_k1_KO_s103
                                 + 0.807882 w_k1_MAT_s103
                                 + 0.758846 w_k1_MSFT_s103
                                 + 0.732738 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.06838 w_k1_ADM_s103
                                 + 0.74078 w_k1_AXP_s103
                                 + 0.947664 w_k1_DIS_s103
                                 + 0.83774 w_k1_FDX_s103
                                 + 0.982457 w_k1_IBM_s103
                                 + 0.824832 w_k1_INTC_s103
                                 + 0.965324 w_k1_KO_s103
                                 + 0.733294 w_k1_MAT_s103
                                 + 0.833872 w_k1_MSFT_s103
                                 + 0.95742 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.23385 w_k1_ADM_s104
                                 + 1.04698 w_k1_AXP_s104
                                 + 1.15596 w_k1_DIS_s104
                                 + 1.21019 w_k1_FDX_s104
                                 + 0.944109 w_k1_IBM_s104
                                 + 1.22377 w_k1_INTC_s104
                                 + 0.958012 w_k1_KO_s104
                                 + 0.731189 w_k1_MAT_s104
                                 + 1.02322 w_k1_MSFT_s104
                                 + 1.01965 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.892996 w_k1_ADM_s104
                                 + 0.942813 w_k1_AXP_s104
                                 + 0.8889 w_k1_DIS_s104
                                 + 0.928854 w_k1_FDX_s104
                                 + 1.16234 w_k1_IBM_s104
                                 + 1.12709 w_k1_INTC_s104
                                 + 1.02253 w_k1_KO_s104 + 1.31132 w_k1_MAT_s104
                                 + 0.942117 w_k1_MSFT_s104
                                 + 0.960079 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.04178 w_k1_ADM_s104
                                 + 0.870106 w_k1_AXP_s104
                                 + 0.762852 w_k1_DIS_s104
                                 + 0.940769 w_k1_FDX_s104
                                 + 0.813432 w_k1_IBM_s104
                                 + 1.07994 w_k1_INTC_s104
                                 + 0.778112 w_k1_KO_s104
                                 + 0.559705 w_k1_MAT_s104
                                 + 0.991965 w_k1_MSFT_s104
                                 + 0.98974 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.780291 w_k1_ADM_s104
                                 + 0.593872 w_k1_AXP_s104
                                 + 0.57179 w_k1_DIS_s104
                                 + 0.82092 w_k1_FDX_s104
                                 + 0.478088 w_k1_IBM_s104
                                 + 0.934876 w_k1_INTC_s104
                                 + 0.624379 w_k1_KO_s104
                                 + 0.478485 w_k1_MAT_s104
                                 + 0.914417 w_k1_MSFT_s104
                                 + 0.839488 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.03286 w_k1_ADM_s104
                                 + 0.751227 w_k1_AXP_s104
                                 + 0.53105 w_k1_DIS_s104
                                 + 0.790434 w_k1_FDX_s104
                                 + 0.696066 w_k1_IBM_s104
                                 + 0.700575 w_k1_INTC_s104
                                 + 0.574133 w_k1_KO_s104
                                 + 0.728638 w_k1_MAT_s104
                                 + 0.754979 w_k1_MSFT_s104
                                 + 0.782437 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.64261 w_k1_ADM_s104
                                 + 1.02864 w_k1_AXP_s104
                                 + 1.01059 w_k1_DIS_s104
                                 + 0.997059 w_k1_FDX_s104
                                 + 1.52633 w_k1_IBM_s104
                                 + 1.29513 w_k1_INTC_s104
                                 + 1.08897 w_k1_KO_s104 + 1.5919 w_k1_MAT_s104
                                 + 1.01384 w_k1_MSFT_s104
                                 + 1.02681 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.24743 w_k1_ADM_s104
                                 + 1.63351 w_k1_AXP_s104
                                 + 1.12839 w_k1_DIS_s104
                                 + 1.40607 w_k1_FDX_s104
                                 + 1.04995 w_k1_IBM_s104
                                 + 1.26988 w_k1_INTC_s104
                                 + 0.961413 w_k1_KO_s104
                                 + 1.29554 w_k1_MAT_s104
                                 + 1.11582 w_k1_MSFT_s104 + 1.0492 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.40416 w_k1_ADM_s104
                                 + 1.70646 w_k1_AXP_s104
                                 + 1.61579 w_k1_DIS_s104
                                 + 1.53851 w_k1_FDX_s104
                                 + 1.30066 w_k1_IBM_s104
                                 + 1.43585 w_k1_INTC_s104
                                 + 1.26164 w_k1_KO_s104 + 1.40226 w_k1_MAT_s104
                                 + 1.23654 w_k1_MSFT_s104
                                 + 1.14526 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.24539 w_k1_ADM_s105
                                 + 0.958717 w_k1_AXP_s105
                                 + 1.07056 w_k1_DIS_s105
                                 + 0.993449 w_k1_FDX_s105
                                 + 1.09771 w_k1_IBM_s105
                                 + 0.908241 w_k1_INTC_s105
                                 + 0.984092 w_k1_KO_s105
                                 + 1.09215 w_k1_MAT_s105
                                 + 1.04188 w_k1_MSFT_s105
                                 + 0.985373 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.12927 w_k1_ADM_s105
                                 + 1.04006 w_k1_AXP_s105
                                 + 1.00526 w_k1_DIS_s105
                                 + 0.977737 w_k1_FDX_s105
                                 + 0.870888 w_k1_IBM_s105
                                 + 0.860165 w_k1_INTC_s105
                                 + 0.930018 w_k1_KO_s105
                                 + 0.759026 w_k1_MAT_s105
                                 + 1.03484 w_k1_MSFT_s105
                                 + 0.969892 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.30783 w_k1_ADM_s105
                                 + 1.12419 w_k1_AXP_s105
                                 + 1.51614 w_k1_DIS_s105
                                 + 1.07442 w_k1_FDX_s105
                                 + 1.77222 w_k1_IBM_s105
                                 + 1.01273 w_k1_INTC_s105
                                 + 1.53488 w_k1_KO_s105 + 1.51338 w_k1_MAT_s105
                                 + 1.35553 w_k1_MSFT_s105
                                 + 1.17751 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.32057 w_k1_ADM_s105
                                 + 1.33215 w_k1_AXP_s105
                                 + 1.09964 w_k1_DIS_s105
                                 + 1.01444 w_k1_FDX_s105
                                 + 1.32366 w_k1_IBM_s105
                                 + 1.06429 w_k1_INTC_s105
                                 + 1.19326 w_k1_KO_s105 + 1.27156 w_k1_MAT_s105
                                 + 1.01496 w_k1_MSFT_s105
                                 + 0.958141 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.999051 w_k1_ADM_s105
                                 + 0.686724 w_k1_AXP_s105
                                 + 0.723072 w_k1_DIS_s105
                                 + 0.846533 w_k1_FDX_s105
                                 + 0.727568 w_k1_IBM_s105
                                 + 0.808245 w_k1_INTC_s105
                                 + 0.770183 w_k1_KO_s105
                                 + 0.627563 w_k1_MAT_s105
                                 + 0.962067 w_k1_MSFT_s105
                                 + 0.845368 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.00652 w_k1_ADM_s105
                                 + 1.63183 w_k1_AXP_s105
                                 + 1.29088 w_k1_DIS_s105
                                 + 1.18266 w_k1_FDX_s105
                                 + 1.42712 w_k1_IBM_s105
                                 + 0.875258 w_k1_INTC_s105
                                 + 1.20877 w_k1_KO_s105 + 2.27577 w_k1_MAT_s105
                                 + 1.1055 w_k1_MSFT_s105 + 1.14085 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.68505 w_k1_ADM_s105
                                 + 1.47611 w_k1_AXP_s105
                                 + 1.33672 w_k1_DIS_s105
                                 + 1.39464 w_k1_FDX_s105
                                 + 0.960768 w_k1_IBM_s105
                                 + 0.953803 w_k1_INTC_s105
                                 + 1.1342 w_k1_KO_s105 + 1.27548 w_k1_MAT_s105
                                 + 1.17228 w_k1_MSFT_s105
                                 + 1.06354 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.987449 w_k1_ADM_s105
                                 + 0.893329 w_k1_AXP_s105
                                 + 0.774669 w_k1_DIS_s105
                                 + 0.795966 w_k1_FDX_s105
                                 + 1.25885 w_k1_IBM_s105
                                 + 0.640751 w_k1_INTC_s105
                                 + 0.979431 w_k1_KO_s105
                                 + 1.45449 w_k1_MAT_s105
                                 + 0.81837 w_k1_MSFT_s105
                                 + 1.01738 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.1593 w_k1_ADM_s106
                                 + 1.26746 w_k1_AXP_s106
                                 + 1.55223 w_k1_DIS_s106 + 1.2151 w_k1_FDX_s106
                                 + 1.41754 w_k1_IBM_s106
                                 + 1.00363 w_k1_INTC_s106
                                 + 1.15388 w_k1_KO_s106 + 1.34556 w_k1_MAT_s106
                                 + 1.14435 w_k1_MSFT_s106
                                 + 1.18193 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02232 w_k1_ADM_s106
                                 + 1.13249 w_k1_AXP_s106 + 1.2102 w_k1_DIS_s106
                                 + 1.08978 w_k1_FDX_s106
                                 + 0.875112 w_k1_IBM_s106
                                 + 1.08892 w_k1_INTC_s106
                                 + 0.817144 w_k1_KO_s106
                                 + 1.04914 w_k1_MAT_s106
                                 + 1.03996 w_k1_MSFT_s106
                                 + 0.953893 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.973159 w_k1_ADM_s106
                                 + 0.881066 w_k1_AXP_s106
                                 + 0.88293 w_k1_DIS_s106
                                 + 0.985017 w_k1_FDX_s106
                                 + 1.05032 w_k1_IBM_s106
                                 + 0.96007 w_k1_INTC_s106
                                 + 0.837899 w_k1_KO_s106
                                 + 1.11345 w_k1_MAT_s106
                                 + 1.00092 w_k1_MSFT_s106
                                 + 0.992061 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.06135 w_k1_ADM_s106
                                 + 0.926494 w_k1_AXP_s106
                                 + 0.736928 w_k1_DIS_s106
                                 + 0.972698 w_k1_FDX_s106
                                 + 0.748262 w_k1_IBM_s106
                                 + 0.758568 w_k1_INTC_s106
                                 + 0.663462 w_k1_KO_s106
                                 + 0.935189 w_k1_MAT_s106
                                 + 0.964171 w_k1_MSFT_s106
                                 + 0.82516 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.785845 w_k1_ADM_s106
                                 + 0.625458 w_k1_AXP_s106
                                 + 0.735161 w_k1_DIS_s106
                                 + 0.719296 w_k1_FDX_s106
                                 + 0.711103 w_k1_IBM_s106
                                 + 0.619507 w_k1_INTC_s106
                                 + 0.638575 w_k1_KO_s106
                                 + 0.909657 w_k1_MAT_s106
                                 + 0.837988 w_k1_MSFT_s106
                                 + 0.743362 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.7836 w_k1_ADM_s106
                                 + 0.932455 w_k1_AXP_s106
                                 + 0.894183 w_k1_DIS_s106
                                 + 0.786877 w_k1_FDX_s106
                                 + 1.25827 w_k1_IBM_s106 + 1.022 w_k1_INTC_s106
                                 + 0.953236 w_k1_KO_s106
                                 + 1.39923 w_k1_MAT_s106
                                 + 1.0222 w_k1_MSFT_s106
                                 + 0.871839 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.15427 w_k1_ADM_s106
                                 + 1.27852 w_k1_AXP_s106
                                 + 1.31166 w_k1_DIS_s106
                                 + 1.24442 w_k1_FDX_s106
                                 + 1.08907 w_k1_IBM_s106
                                 + 1.21917 w_k1_INTC_s106
                                 + 1.13367 w_k1_KO_s106 + 1.42353 w_k1_MAT_s106
                                 + 1.32815 w_k1_MSFT_s106
                                 + 1.09006 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.903099 w_k1_ADM_s106
                                 + 0.662648 w_k1_AXP_s106
                                 + 0.920707 w_k1_DIS_s106
                                 + 0.920455 w_k1_FDX_s106
                                 + 0.855212 w_k1_IBM_s106
                                 + 0.914693 w_k1_INTC_s106
                                 + 0.825188 w_k1_KO_s106
                                 + 0.913176 w_k1_MAT_s106
                                 + 1.00488 w_k1_MSFT_s106
                                 + 0.874688 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.95957 w_k1_ADM_s107
                                 + 0.968434 w_k1_AXP_s107
                                 + 1.04289 w_k1_DIS_s107
                                 + 1.11005 w_k1_FDX_s107
                                 + 0.858894 w_k1_IBM_s107
                                 + 1.04787 w_k1_INTC_s107
                                 + 0.861466 w_k1_KO_s107
                                 + 0.580752 w_k1_MAT_s107
                                 + 0.811639 w_k1_MSFT_s107
                                 + 0.908528 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.09261 w_k1_ADM_s107
                                 + 1.08025 w_k1_AXP_s107
                                 + 1.08597 w_k1_DIS_s107
                                 + 1.23162 w_k1_FDX_s107
                                 + 0.717967 w_k1_IBM_s107
                                 + 1.1922 w_k1_INTC_s107
                                 + 0.868759 w_k1_KO_s107
                                 + 0.81296 w_k1_MAT_s107
                                 + 1.12538 w_k1_MSFT_s107
                                 + 1.06473 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.27253 w_k1_ADM_s107
                                 + 1.27405 w_k1_AXP_s107
                                 + 1.66505 w_k1_DIS_s107
                                 + 1.25072 w_k1_FDX_s107
                                 + 1.16736 w_k1_IBM_s107
                                 + 1.04111 w_k1_INTC_s107
                                 + 1.35137 w_k1_KO_s107 + 1.26117 w_k1_MAT_s107
                                 + 1.14378 w_k1_MSFT_s107
                                 + 1.14043 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.18691 w_k1_ADM_s107
                                 + 1.33038 w_k1_AXP_s107 + 1.0318 w_k1_DIS_s107
                                 + 0.969302 w_k1_FDX_s107
                                 + 1.13506 w_k1_IBM_s107
                                 + 1.05929 w_k1_INTC_s107
                                 + 1.09677 w_k1_KO_s107 + 1.12355 w_k1_MAT_s107
                                 + 0.921572 w_k1_MSFT_s107
                                 + 1.13348 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.792407 w_k1_ADM_s107
                                 + 0.701096 w_k1_AXP_s107
                                 + 0.579599 w_k1_DIS_s107
                                 + 0.688068 w_k1_FDX_s107
                                 + 0.83875 w_k1_IBM_s107
                                 + 0.745362 w_k1_INTC_s107
                                 + 0.764983 w_k1_KO_s107
                                 + 0.804147 w_k1_MAT_s107
                                 + 0.724025 w_k1_MSFT_s107
                                 + 0.814844 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.08309 w_k1_ADM_s107
                                 + 1.42818 w_k1_AXP_s107
                                 + 1.27424 w_k1_DIS_s107
                                 + 1.21528 w_k1_FDX_s107
                                 + 1.44337 w_k1_IBM_s107
                                 + 1.20475 w_k1_INTC_s107
                                 + 1.21659 w_k1_KO_s107 + 1.73119 w_k1_MAT_s107
                                 + 1.07077 w_k1_MSFT_s107
                                 + 1.23693 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.839874 w_k1_ADM_s107
                                 + 0.850509 w_k1_AXP_s107
                                 + 1.01337 w_k1_DIS_s107
                                 + 0.864808 w_k1_FDX_s107
                                 + 1.09407 w_k1_IBM_s107
                                 + 1.12514 w_k1_INTC_s107
                                 + 1.05045 w_k1_KO_s107 + 1.01572 w_k1_MAT_s107
                                 + 0.886896 w_k1_MSFT_s107
                                 + 1.03361 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.819037 w_k1_ADM_s107
                                 + 1.18917 w_k1_AXP_s107
                                 + 0.776132 w_k1_DIS_s107
                                 + 0.874504 w_k1_FDX_s107
                                 + 0.997743 w_k1_IBM_s107
                                 + 0.907972 w_k1_INTC_s107
                                 + 0.882575 w_k1_KO_s107
                                 + 1.06313 w_k1_MAT_s107
                                 + 0.753206 w_k1_MSFT_s107
                                 + 0.990538 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.997907 w_k1_ADM_s110
                                 + 0.888986 w_k1_AXP_s110
                                 + 0.831804 w_k1_DIS_s110
                                 + 0.89628 w_k1_FDX_s110
                                 + 1.11046 w_k1_IBM_s110
                                 + 0.962465 w_k1_INTC_s110
                                 + 0.972134 w_k1_KO_s110
                                 + 0.892123 w_k1_MAT_s110
                                 + 0.883732 w_k1_MSFT_s110
                                 + 1.14013 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.52032 w_k1_ADM_s110
                                 + 1.45421 w_k1_AXP_s110
                                 + 1.74335 w_k1_DIS_s110
                                 + 1.38358 w_k1_FDX_s110
                                 + 1.80848 w_k1_IBM_s110
                                 + 1.29596 w_k1_INTC_s110
                                 + 1.66123 w_k1_KO_s110 + 1.66435 w_k1_MAT_s110
                                 + 1.2925 w_k1_MSFT_s110 + 1.25802 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.710285 w_k1_ADM_s110
                                 + 0.654044 w_k1_AXP_s110
                                 + 0.587514 w_k1_DIS_s110
                                 + 0.726663 w_k1_FDX_s110
                                 + 0.645391 w_k1_IBM_s110
                                 + 0.877967 w_k1_INTC_s110
                                 + 0.674123 w_k1_KO_s110
                                 + 0.504372 w_k1_MAT_s110
                                 + 0.726974 w_k1_MSFT_s110
                                 + 0.767812 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.915979 w_k1_ADM_s110
                                 + 0.888791 w_k1_AXP_s110
                                 + 0.652841 w_k1_DIS_s110
                                 + 0.911684 w_k1_FDX_s110
                                 + 0.61434 w_k1_IBM_s110
                                 + 1.00776 w_k1_INTC_s110
                                 + 0.675711 w_k1_KO_s110
                                 + 0.531169 w_k1_MAT_s110
                                 + 0.785564 w_k1_MSFT_s110
                                 + 0.911512 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.966648 w_k1_ADM_s110
                                 + 0.843689 w_k1_AXP_s110
                                 + 0.834755 w_k1_DIS_s110
                                 + 1.06402 w_k1_FDX_s110
                                 + 0.72253 w_k1_IBM_s110
                                 + 1.23157 w_k1_INTC_s110
                                 + 0.85473 w_k1_KO_s110
                                 + 0.586724 w_k1_MAT_s110
                                 + 0.999784 w_k1_MSFT_s110
                                 + 1.04182 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.0663 w_k1_ADM_s110
                                 + 1.29611 w_k1_AXP_s110
                                 + 0.853007 w_k1_DIS_s110
                                 + 1.01491 w_k1_FDX_s110
                                 + 0.821236 w_k1_IBM_s110
                                 + 0.982219 w_k1_INTC_s110
                                 + 0.830146 w_k1_KO_s110
                                 + 0.898913 w_k1_MAT_s110
                                 + 0.98584 w_k1_MSFT_s110
                                 + 0.938798 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.0972 w_k1_ADM_s110
                                 + 1.32631 w_k1_AXP_s110
                                 + 1.16548 w_k1_DIS_s110
                                 + 1.08912 w_k1_FDX_s110
                                 + 1.37899 w_k1_IBM_s110
                                 + 1.0461 w_k1_INTC_s110 + 1.26283 w_k1_KO_s110
                                 + 1.49727 w_k1_MAT_s110
                                 + 0.915649 w_k1_MSFT_s110
                                 + 1.2912 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.2145 w_k1_ADM_s110
                                 + 1.28008 w_k1_AXP_s110
                                 + 1.34406 w_k1_DIS_s110
                                 + 1.29718 w_k1_FDX_s110
                                 + 0.933493 w_k1_IBM_s110
                                 + 1.02477 w_k1_INTC_s110
                                 + 1.11253 w_k1_KO_s110 + 1.03306 w_k1_MAT_s110
                                 + 1.07352 w_k1_MSFT_s110
                                 + 1.19585 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.22668 w_k1_ADM_s111
                                 + 1.26388 w_k1_AXP_s111
                                 + 1.28724 w_k1_DIS_s111
                                 + 1.44506 w_k1_FDX_s111
                                 + 0.709141 w_k1_IBM_s111
                                 + 1.21469 w_k1_INTC_s111
                                 + 0.837829 w_k1_KO_s111
                                 + 0.892225 w_k1_MAT_s111
                                 + 1.11098 w_k1_MSFT_s111
                                 + 1.04175 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.02168 w_k1_ADM_s111
                                 + 0.896471 w_k1_AXP_s111
                                 + 0.882113 w_k1_DIS_s111
                                 + 0.852141 w_k1_FDX_s111
                                 + 1.36494 w_k1_IBM_s111
                                 + 1.05747 w_k1_INTC_s111
                                 + 1.05634 w_k1_KO_s111
                                 + 0.957108 w_k1_MAT_s111
                                 + 0.919567 w_k1_MSFT_s111
                                 + 1.07728 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.10277 w_k1_ADM_s111
                                 + 1.26791 w_k1_AXP_s111
                                 + 1.24323 w_k1_DIS_s111
                                 + 1.19476 w_k1_FDX_s111
                                 + 1.57339 w_k1_IBM_s111
                                 + 1.06596 w_k1_INTC_s111 + 1.1718 w_k1_KO_s111
                                 + 1.43553 w_k1_MAT_s111
                                 + 1.09503 w_k1_MSFT_s111 + 1.1802 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.03814 w_k1_ADM_s111
                                 + 1.12116 w_k1_AXP_s111
                                 + 0.812383 w_k1_DIS_s111
                                 + 0.998315 w_k1_FDX_s111
                                 + 1.04071 w_k1_IBM_s111
                                 + 0.94012 w_k1_INTC_s111
                                 + 0.826795 w_k1_KO_s111
                                 + 0.887492 w_k1_MAT_s111
                                 + 0.783861 w_k1_MSFT_s111
                                 + 0.956971 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.76251 w_k1_ADM_s111
                                 + 0.658765 w_k1_AXP_s111
                                 + 0.613667 w_k1_DIS_s111
                                 + 0.809396 w_k1_FDX_s111
                                 + 0.81536 w_k1_IBM_s111
                                 + 0.777874 w_k1_INTC_s111
                                 + 0.757938 w_k1_KO_s111
                                 + 0.756223 w_k1_MAT_s111
                                 + 0.833646 w_k1_MSFT_s111
                                 + 0.844355 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.931682 w_k1_ADM_s111
                                 + 1.37793 w_k1_AXP_s111
                                 + 1.32771 w_k1_DIS_s111
                                 + 1.25558 w_k1_FDX_s111
                                 + 1.17668 w_k1_IBM_s111
                                 + 1.21935 w_k1_INTC_s111
                                 + 1.05026 w_k1_KO_s111 + 1.34459 w_k1_MAT_s111
                                 + 1.1589 w_k1_MSFT_s111 + 1.22016 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.918627 w_k1_ADM_s111
                                 + 0.876589 w_k1_AXP_s111
                                 + 0.918534 w_k1_DIS_s111
                                 + 0.989889 w_k1_FDX_s111
                                 + 0.852449 w_k1_IBM_s111
                                 + 0.905849 w_k1_INTC_s111
                                 + 0.950534 w_k1_KO_s111
                                 + 0.983854 w_k1_MAT_s111
                                 + 0.953522 w_k1_MSFT_s111
                                 + 1.08715 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.08792 w_k1_ADM_s111
                                 + 0.818317 w_k1_AXP_s111
                                 + 1.18735 w_k1_DIS_s111
                                 + 1.17805 w_k1_FDX_s111
                                 + 0.911324 w_k1_IBM_s111
                                 + 0.983872 w_k1_INTC_s111
                                 + 0.901603 w_k1_KO_s111
                                 + 0.775566 w_k1_MAT_s111
                                 + 1.19664 w_k1_MSFT_s111
                                 + 1.05872 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.20788 w_k1_ADM_s112
                                 + 1.53393 w_k1_AXP_s112
                                 + 1.75185 w_k1_DIS_s112
                                 + 1.41436 w_k1_FDX_s112
                                 + 1.13354 w_k1_IBM_s112
                                 + 1.57915 w_k1_INTC_s112
                                 + 1.26158 w_k1_KO_s112 + 1.00604 w_k1_MAT_s112
                                 + 1.43483 w_k1_MSFT_s112
                                 + 1.11809 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.04335 w_k1_ADM_s112
                                 + 1.09276 w_k1_AXP_s112
                                 + 1.05697 w_k1_DIS_s112 + 0.9695 w_k1_FDX_s112
                                 + 1.03897 w_k1_IBM_s112
                                 + 1.18135 w_k1_INTC_s112
                                 + 1.06786 w_k1_KO_s112
                                 + 0.965192 w_k1_MAT_s112
                                 + 1.0707 w_k1_MSFT_s112 + 1.00546 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.08507 w_k1_ADM_s112
                                 + 1.34233 w_k1_AXP_s112
                                 + 1.37068 w_k1_DIS_s112
                                 + 1.28574 w_k1_FDX_s112
                                 + 1.22106 w_k1_IBM_s112 + 1.443 w_k1_INTC_s112
                                 + 1.14908 w_k1_KO_s112 + 1.31431 w_k1_MAT_s112
                                 + 1.22736 w_k1_MSFT_s112
                                 + 1.08162 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.985149 w_k1_ADM_s112
                                 + 0.8856 w_k1_AXP_s112 + 1.49439 w_k1_DIS_s112
                                 + 1.0967 w_k1_FDX_s112 + 1.15444 w_k1_IBM_s112
                                 + 1.03197 w_k1_INTC_s112
                                 + 1.22298 w_k1_KO_s112 + 1.18312 w_k1_MAT_s112
                                 + 1.2782 w_k1_MSFT_s112 + 1.06244 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.05694 w_k1_ADM_s112
                                 + 0.804504 w_k1_AXP_s112
                                 + 0.981474 w_k1_DIS_s112
                                 + 0.902639 w_k1_FDX_s112
                                 + 1.36525 w_k1_IBM_s112
                                 + 1.01673 w_k1_INTC_s112
                                 + 1.19806 w_k1_KO_s112 + 1.44408 w_k1_MAT_s112
                                 + 1.07498 w_k1_MSFT_s112
                                 + 0.968019 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.21825 w_k1_ADM_s112
                                 + 1.10408 w_k1_AXP_s112 + 1.2377 w_k1_DIS_s112
                                 + 1.31754 w_k1_FDX_s112
                                 + 0.70983 w_k1_IBM_s112
                                 + 1.38877 w_k1_INTC_s112
                                 + 1.01319 w_k1_KO_s112
                                 + 0.677888 w_k1_MAT_s112
                                 + 1.22847 w_k1_MSFT_s112
                                 + 1.07562 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.671721 w_k1_ADM_s112
                                 + 0.650126 w_k1_AXP_s112
                                 + 0.714732 w_k1_DIS_s112
                                 + 0.73457 w_k1_FDX_s112
                                 + 0.84872 w_k1_IBM_s112
                                 + 0.81451 w_k1_INTC_s112
                                 + 0.754685 w_k1_KO_s112
                                 + 0.714393 w_k1_MAT_s112
                                 + 0.810965 w_k1_MSFT_s112
                                 + 0.739828 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.853681 w_k1_ADM_s112
                                 + 0.838505 w_k1_AXP_s112
                                 + 0.89796 w_k1_DIS_s112
                                 + 0.921464 w_k1_FDX_s112
                                 + 0.814759 w_k1_IBM_s112
                                 + 0.985369 w_k1_INTC_s112
                                 + 0.910312 w_k1_KO_s112
                                 + 0.961721 w_k1_MAT_s112
                                 + 0.997921 w_k1_MSFT_s112
                                 + 0.813722 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.15677 w_k1_ADM_s113
                                 + 1.29657 w_k1_AXP_s113
                                 + 1.66999 w_k1_DIS_s113
                                 + 1.35071 w_k1_FDX_s113
                                 + 1.60312 w_k1_IBM_s113
                                 + 1.12598 w_k1_INTC_s113
                                 + 1.40486 w_k1_KO_s113 + 2.05528 w_k1_MAT_s113
                                 + 1.3948 w_k1_MSFT_s113 + 1.27968 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.15558 w_k1_ADM_s113
                                 + 1.26879 w_k1_AXP_s113
                                 + 1.39719 w_k1_DIS_s113
                                 + 1.46494 w_k1_FDX_s113
                                 + 0.769687 w_k1_IBM_s113
                                 + 1.34817 w_k1_INTC_s113
                                 + 1.06653 w_k1_KO_s113
                                 + 0.924928 w_k1_MAT_s113
                                 + 1.35522 w_k1_MSFT_s113
                                 + 1.11059 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.732634 w_k1_ADM_s113
                                 + 0.811036 w_k1_AXP_s113
                                 + 0.718131 w_k1_DIS_s113
                                 + 0.812447 w_k1_FDX_s113
                                 + 1.1088 w_k1_IBM_s113
                                 + 0.749793 w_k1_INTC_s113
                                 + 0.864607 w_k1_KO_s113
                                 + 1.34487 w_k1_MAT_s113
                                 + 0.857595 w_k1_MSFT_s113
                                 + 0.99187 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.06025 w_k1_ADM_s113
                                 + 0.945035 w_k1_AXP_s113
                                 + 0.951619 w_k1_DIS_s113
                                 + 0.918388 w_k1_FDX_s113
                                 + 1.36593 w_k1_IBM_s113
                                 + 0.924388 w_k1_INTC_s113
                                 + 1.10057 w_k1_KO_s113 + 1.48861 w_k1_MAT_s113
                                 + 0.891956 w_k1_MSFT_s113
                                 + 1.16277 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.937816 w_k1_ADM_s113
                                 + 0.858977 w_k1_AXP_s113
                                 + 0.747361 w_k1_DIS_s113
                                 + 0.861713 w_k1_FDX_s113
                                 + 0.810657 w_k1_IBM_s113
                                 + 0.726492 w_k1_INTC_s113
                                 + 0.767786 w_k1_KO_s113
                                 + 0.892165 w_k1_MAT_s113
                                 + 0.837615 w_k1_MSFT_s113
                                 + 0.942306 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.04662 w_k1_ADM_s113
                                 + 1.02715 w_k1_AXP_s113
                                 + 0.940181 w_k1_DIS_s113
                                 + 1.02446 w_k1_FDX_s113
                                 + 0.923539 w_k1_IBM_s113
                                 + 1.09416 w_k1_INTC_s113
                                 + 0.970769 w_k1_KO_s113
                                 + 1.10131 w_k1_MAT_s113
                                 + 1.07495 w_k1_MSFT_s113
                                 + 1.02933 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.635225 w_k1_ADM_s113
                                 + 0.663571 w_k1_AXP_s113
                                 + 0.538543 w_k1_DIS_s113
                                 + 0.698091 w_k1_FDX_s113
                                 + 0.684515 w_k1_IBM_s113
                                 + 0.687034 w_k1_INTC_s113
                                 + 0.687975 w_k1_KO_s113
                                 + 0.886512 w_k1_MAT_s113
                                 + 0.714148 w_k1_MSFT_s113
                                 + 0.794701 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.11728 w_k1_ADM_s113
                                 + 1.32282 w_k1_AXP_s113
                                 + 1.27831 w_k1_DIS_s113
                                 + 1.28222 w_k1_FDX_s113
                                 + 1.06569 w_k1_IBM_s113
                                 + 1.10843 w_k1_INTC_s113
                                 + 1.25288 w_k1_KO_s113 + 1.63382 w_k1_MAT_s113
                                 + 1.24333 w_k1_MSFT_s113
                                 + 1.25009 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.32288 w_k1_ADM_s114
                                 + 1.02371 w_k1_AXP_s114
                                 + 1.19249 w_k1_DIS_s114
                                 + 1.19216 w_k1_FDX_s114
                                 + 0.816832 w_k1_IBM_s114
                                 + 1.07345 w_k1_INTC_s114
                                 + 0.993235 w_k1_KO_s114
                                 + 0.756853 w_k1_MAT_s114
                                 + 1.13436 w_k1_MSFT_s114
                                 + 1.16261 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.988257 w_k1_ADM_s114
                                 + 0.735281 w_k1_AXP_s114
                                 + 0.690873 w_k1_DIS_s114
                                 + 0.77171 w_k1_FDX_s114
                                 + 1.07031 w_k1_IBM_s114
                                 + 0.877807 w_k1_INTC_s114
                                 + 0.960304 w_k1_KO_s114
                                 + 0.896831 w_k1_MAT_s114
                                 + 0.867486 w_k1_MSFT_s114
                                 + 0.964382 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.2801 w_k1_ADM_s114
                                 + 1.29102 w_k1_AXP_s114 + 1.3341 w_k1_DIS_s114
                                 + 1.14293 w_k1_FDX_s114
                                 + 1.30326 w_k1_IBM_s114
                                 + 1.32987 w_k1_INTC_s114
                                 + 1.15149 w_k1_KO_s114 + 1.26001 w_k1_MAT_s114
                                 + 1.15076 w_k1_MSFT_s114
                                 + 1.36341 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.18078 w_k1_ADM_s114
                                 + 1.37187 w_k1_AXP_s114
                                 + 0.911225 w_k1_DIS_s114
                                 + 1.18142 w_k1_FDX_s114
                                 + 0.750338 w_k1_IBM_s114
                                 + 1.19064 w_k1_INTC_s114
                                 + 0.848509 w_k1_KO_s114
                                 + 0.837371 w_k1_MAT_s114
                                 + 0.90233 w_k1_MSFT_s114
                                 + 0.956481 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.15426 w_k1_ADM_s114
                                 + 1.0103 w_k1_AXP_s114 + 1.06753 w_k1_DIS_s114
                                 + 1.06513 w_k1_FDX_s114 + 1.0678 w_k1_IBM_s114
                                 + 0.948057 w_k1_INTC_s114
                                 + 1.08747 w_k1_KO_s114 + 1.53171 w_k1_MAT_s114
                                 + 1.08708 w_k1_MSFT_s114
                                 + 1.14413 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.917702 w_k1_ADM_s114
                                 + 1.42941 w_k1_AXP_s114
                                 + 1.17373 w_k1_DIS_s114
                                 + 0.936808 w_k1_FDX_s114
                                 + 1.7795 w_k1_IBM_s114
                                 + 0.950006 w_k1_INTC_s114
                                 + 1.30621 w_k1_KO_s114 + 1.98768 w_k1_MAT_s114
                                 + 0.962537 w_k1_MSFT_s114
                                 + 1.12663 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.936814 w_k1_ADM_s114
                                 + 0.994034 w_k1_AXP_s114
                                 + 0.851308 w_k1_DIS_s114
                                 + 0.938177 w_k1_FDX_s114
                                 + 0.912832 w_k1_IBM_s114
                                 + 0.986744 w_k1_INTC_s114
                                 + 0.87432 w_k1_KO_s114
                                 + 0.915468 w_k1_MAT_s114
                                 + 0.850401 w_k1_MSFT_s114
                                 + 1.06293 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.909701 w_k1_ADM_s114
                                 + 0.747661 w_k1_AXP_s114
                                 + 0.653655 w_k1_DIS_s114
                                 + 0.803929 w_k1_FDX_s114
                                 + 0.680309 w_k1_IBM_s114
                                 + 0.915588 w_k1_INTC_s114
                                 + 0.754853 w_k1_KO_s114
                                 + 0.627561 w_k1_MAT_s114
                                 + 0.842108 w_k1_MSFT_s114
                                 + 0.983068 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17322 w_k1_ADM_s115
                                 + 0.985145 w_k1_AXP_s115
                                 + 1.1431 w_k1_DIS_s115 + 1.10709 w_k1_FDX_s115
                                 + 0.897433 w_k1_IBM_s115
                                 + 1.24072 w_k1_INTC_s115
                                 + 1.08463 w_k1_KO_s115
                                 + 0.781559 w_k1_MAT_s115
                                 + 1.06987 w_k1_MSFT_s115
                                 + 1.12858 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.888822 w_k1_ADM_s115
                                 + 1.10074 w_k1_AXP_s115
                                 + 1.00334 w_k1_DIS_s115
                                 + 0.949019 w_k1_FDX_s115
                                 + 1.42056 w_k1_IBM_s115
                                 + 1.24548 w_k1_INTC_s115
                                 + 1.14048 w_k1_KO_s115 + 1.13661 w_k1_MAT_s115
                                 + 1.08913 w_k1_MSFT_s115
                                 + 1.12054 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.25112 w_k1_ADM_s115
                                 + 1.35662 w_k1_AXP_s115
                                 + 1.09928 w_k1_DIS_s115 + 1.1454 w_k1_FDX_s115
                                 + 0.933563 w_k1_IBM_s115
                                 + 1.14764 w_k1_INTC_s115
                                 + 0.998069 w_k1_KO_s115
                                 + 0.798045 w_k1_MAT_s115
                                 + 1.11058 w_k1_MSFT_s115
                                 + 1.04655 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.901046 w_k1_ADM_s115
                                 + 0.849945 w_k1_AXP_s115
                                 + 0.833491 w_k1_DIS_s115
                                 + 0.971459 w_k1_FDX_s115
                                 + 0.725103 w_k1_IBM_s115
                                 + 1.13349 w_k1_INTC_s115
                                 + 0.795771 w_k1_KO_s115
                                 + 0.487002 w_k1_MAT_s115
                                 + 0.93897 w_k1_MSFT_s115
                                 + 0.954697 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04024 w_k1_ADM_s115
                                 + 1.13072 w_k1_AXP_s115
                                 + 0.862211 w_k1_DIS_s115
                                 + 0.974145 w_k1_FDX_s115
                                 + 1.03841 w_k1_IBM_s115
                                 + 1.05196 w_k1_INTC_s115
                                 + 1.01214 w_k1_KO_s115
                                 + 0.602384 w_k1_MAT_s115
                                 + 0.905054 w_k1_MSFT_s115
                                 + 1.02696 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.32399 w_k1_ADM_s115
                                 + 1.40789 w_k1_AXP_s115
                                 + 1.46679 w_k1_DIS_s115
                                 + 1.32026 w_k1_FDX_s115
                                 + 1.31421 w_k1_IBM_s115
                                 + 1.35716 w_k1_INTC_s115
                                 + 1.19234 w_k1_KO_s115 + 1.14722 w_k1_MAT_s115
                                 + 1.28186 w_k1_MSFT_s115
                                 + 1.21824 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.727643 w_k1_ADM_s115
                                 + 0.936776 w_k1_AXP_s115
                                 + 0.732807 w_k1_DIS_s115
                                 + 0.827284 w_k1_FDX_s115
                                 + 0.923027 w_k1_IBM_s115
                                 + 0.912777 w_k1_INTC_s115
                                 + 0.89734 w_k1_KO_s115
                                 + 0.903518 w_k1_MAT_s115
                                 + 0.837284 w_k1_MSFT_s115
                                 + 0.98415 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.779445 w_k1_ADM_s115
                                 + 0.639876 w_k1_AXP_s115
                                 + 0.602469 w_k1_DIS_s115
                                 + 0.754826 w_k1_FDX_s115
                                 + 0.583574 w_k1_IBM_s115
                                 + 0.807292 w_k1_INTC_s115
                                 + 0.661174 w_k1_KO_s115
                                 + 0.374408 w_k1_MAT_s115
                                 + 0.795644 w_k1_MSFT_s115
                                 + 0.784466 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.0679 w_k1_ADM_s116
                                 + 1.27976 w_k1_AXP_s116
                                 + 1.58746 w_k1_DIS_s116
                                 + 1.14898 w_k1_FDX_s116
                                 + 1.66865 w_k1_IBM_s116
                                 + 1.40777 w_k1_INTC_s116
                                 + 1.51491 w_k1_KO_s116 + 1.58599 w_k1_MAT_s116
                                 + 1.26484 w_k1_MSFT_s116
                                 + 1.28889 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.32836 w_k1_ADM_s116
                                 + 1.26228 w_k1_AXP_s116
                                 + 1.40652 w_k1_DIS_s116
                                 + 1.30685 w_k1_FDX_s116
                                 + 1.05767 w_k1_IBM_s116
                                 + 1.02868 w_k1_INTC_s116 + 1.1158 w_k1_KO_s116
                                 + 1.18892 w_k1_MAT_s116
                                 + 1.10027 w_k1_MSFT_s116
                                 + 1.05869 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.09844 w_k1_ADM_s116
                                 + 0.811818 w_k1_AXP_s116
                                 + 0.8414 w_k1_DIS_s116
                                 + 0.950481 w_k1_FDX_s116
                                 + 0.913061 w_k1_IBM_s116
                                 + 1.02543 w_k1_INTC_s116
                                 + 0.933413 w_k1_KO_s116
                                 + 0.712717 w_k1_MAT_s116
                                 + 0.923355 w_k1_MSFT_s116
                                 + 1.06503 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.953584 w_k1_ADM_s116
                                 + 1.24205 w_k1_AXP_s116
                                 + 1.13842 w_k1_DIS_s116
                                 + 1.07784 w_k1_FDX_s116
                                 + 1.41169 w_k1_IBM_s116
                                 + 1.30329 w_k1_INTC_s116
                                 + 1.03721 w_k1_KO_s116 + 1.47538 w_k1_MAT_s116
                                 + 1.07129 w_k1_MSFT_s116
                                 + 0.969901 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.885349 w_k1_ADM_s116
                                 + 0.767772 w_k1_AXP_s116
                                 + 0.679522 w_k1_DIS_s116
                                 + 0.860111 w_k1_FDX_s116
                                 + 0.646197 w_k1_IBM_s116
                                 + 1.04331 w_k1_INTC_s116
                                 + 0.670178 w_k1_KO_s116
                                 + 0.495894 w_k1_MAT_s116
                                 + 0.872457 w_k1_MSFT_s116
                                 + 0.770057 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.952468 w_k1_ADM_s116
                                 + 0.981759 w_k1_AXP_s116
                                 + 0.764153 w_k1_DIS_s116
                                 + 0.904467 w_k1_FDX_s116
                                 + 1.11855 w_k1_IBM_s116
                                 + 1.11542 w_k1_INTC_s116
                                 + 0.853475 w_k1_KO_s116
                                 + 1.00627 w_k1_MAT_s116
                                 + 0.887898 w_k1_MSFT_s116
                                 + 0.83672 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.1358 w_k1_ADM_s116
                                 + 0.894875 w_k1_AXP_s116
                                 + 0.913637 w_k1_DIS_s116
                                 + 1.05255 w_k1_FDX_s116
                                 + 0.891586 w_k1_IBM_s116
                                 + 1.05463 w_k1_INTC_s116
                                 + 0.894835 w_k1_KO_s116
                                 + 0.519548 w_k1_MAT_s116
                                 + 0.95029 w_k1_MSFT_s116
                                 + 0.974664 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.778333 w_k1_ADM_s116
                                 + 0.713002 w_k1_AXP_s116
                                 + 0.692606 w_k1_DIS_s116
                                 + 0.733786 w_k1_FDX_s116
                                 + 1.04127 w_k1_IBM_s116
                                 + 0.928019 w_k1_INTC_s116
                                 + 0.822574 w_k1_KO_s116
                                 + 0.675015 w_k1_MAT_s116
                                 + 0.758082 w_k1_MSFT_s116
                                 + 0.894817 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.23049 w_k1_ADM_s117
                                 + 1.19863 w_k1_AXP_s117
                                 + 1.19463 w_k1_DIS_s117
                                 + 1.23544 w_k1_FDX_s117
                                 + 0.889091 w_k1_IBM_s117
                                 + 1.2115 w_k1_INTC_s117 + 1.14351 w_k1_KO_s117
                                 + 1.19953 w_k1_MAT_s117
                                 + 1.22025 w_k1_MSFT_s117 + 1.2019 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.11731 w_k1_ADM_s117
                                 + 0.865744 w_k1_AXP_s117
                                 + 1.02066 w_k1_DIS_s117
                                 + 0.997122 w_k1_FDX_s117
                                 + 0.869969 w_k1_IBM_s117
                                 + 0.860131 w_k1_INTC_s117
                                 + 1.0371 w_k1_KO_s117 + 1.03521 w_k1_MAT_s117
                                 + 1.02401 w_k1_MSFT_s117
                                 + 0.958495 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.974594 w_k1_ADM_s117
                                 + 1.57803 w_k1_AXP_s117
                                 + 1.04976 w_k1_DIS_s117
                                 + 1.22209 w_k1_FDX_s117
                                 + 1.04889 w_k1_IBM_s117
                                 + 1.13939 w_k1_INTC_s117
                                 + 1.05752 w_k1_KO_s117 + 1.73833 w_k1_MAT_s117
                                 + 1.00016 w_k1_MSFT_s117
                                 + 1.08153 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15426 w_k1_ADM_s117
                                 + 1.56858 w_k1_AXP_s117 + 1.7666 w_k1_DIS_s117
                                 + 1.38988 w_k1_FDX_s117
                                 + 1.41086 w_k1_IBM_s117
                                 + 1.34621 w_k1_INTC_s117
                                 + 1.56284 w_k1_KO_s117 + 1.62103 w_k1_MAT_s117
                                 + 1.23438 w_k1_MSFT_s117
                                 + 1.08863 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.803175 w_k1_ADM_s117
                                 + 0.729738 w_k1_AXP_s117
                                 + 0.70326 w_k1_DIS_s117
                                 + 0.756313 w_k1_FDX_s117
                                 + 0.699987 w_k1_IBM_s117
                                 + 0.755445 w_k1_INTC_s117
                                 + 0.764119 w_k1_KO_s117
                                 + 0.835464 w_k1_MAT_s117
                                 + 0.879839 w_k1_MSFT_s117
                                 + 0.787595 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.00745 w_k1_ADM_s117
                                 + 1.07145 w_k1_AXP_s117
                                 + 1.07665 w_k1_DIS_s117
                                 + 0.88713 w_k1_FDX_s117
                                 + 1.48978 w_k1_IBM_s117
                                 + 0.936637 w_k1_INTC_s117
                                 + 1.36264 w_k1_KO_s117 + 1.94878 w_k1_MAT_s117
                                 + 0.918661 w_k1_MSFT_s117
                                 + 1.02914 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.980218 w_k1_ADM_s117
                                 + 0.833765 w_k1_AXP_s117
                                 + 0.738873 w_k1_DIS_s117
                                 + 0.81167 w_k1_FDX_s117
                                 + 0.983721 w_k1_IBM_s117
                                 + 0.811968 w_k1_INTC_s117
                                 + 0.784504 w_k1_KO_s117
                                 + 0.94653 w_k1_MAT_s117
                                 + 0.868628 w_k1_MSFT_s117
                                 + 0.89679 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.3256 w_k1_ADM_s117
                                 + 1.16203 w_k1_AXP_s117
                                 + 1.20261 w_k1_DIS_s117
                                 + 1.22316 w_k1_FDX_s117
                                 + 1.28393 w_k1_IBM_s117
                                 + 0.966553 w_k1_INTC_s117
                                 + 1.24511 w_k1_KO_s117 + 1.48067 w_k1_MAT_s117
                                 + 1.15268 w_k1_MSFT_s117
                                 + 1.20861 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0515891507640694 w_k1_ADM_s000
                                 + 0.0303673805336583 w_k1_ADM_s001
                                 + 0.0219074282964711 w_k1_ADM_s002
                                 + 0.0247333331351097 w_k1_ADM_s003
                                 + 0.0538458308759466 w_k1_ADM_s004
                                 + 0.0244278268799204 w_k1_ADM_s005
                                 + 0.0279706794205867 w_k1_ADM_s006
                                 + 0.0522087116816786 w_k1_ADM_s007
                                 + 0.0109196375533484 w_k1_ADM_s010
                                 + 0.038120322295361 w_k1_ADM_s011
                                 + 0.0298921480784963 w_k1_ADM_s012
                                 + 0.0532110543645286 w_k1_ADM_s013
                                 + 0.0134371765894537 w_k1_ADM_s014
                                 + 0.0143953801387864 w_k1_ADM_s015
                                 + 0.0297583728047976 w_k1_ADM_s016
                                 + 0.0358116394290129 w_k1_ADM_s017
                                 + 0.0158346295029377 w_k1_ADM_s100
                                 + 0.0174678521311316 w_k1_ADM_s101
                                 + 0.0245430586952277 w_k1_ADM_s102
                                 + 0.0121314143903179 w_k1_ADM_s103
                                 + 0.0177523365800482 w_k1_ADM_s104
                                 + 0.0183386038044442 w_k1_ADM_s105
                                 + 0.014896242632451 w_k1_ADM_s106
                                 + 0.0216216377675583 w_k1_ADM_s107
                                 + 0.065579108623794 w_k1_ADM_s110
                                 + 0.0655398903685219 w_k1_ADM_s111
                                 + 0.0384603923532131 w_k1_ADM_s112
                                 + 0.0351565113191478 w_k1_ADM_s113
                                 + 0.0566059286349914 w_k1_ADM_s114
                                 + 0.0501903876436381 w_k1_ADM_s115
                                 + 0.0268023664476053 w_k1_ADM_s116
                                 + 0.0284664027109484 w_k1_ADM_s117
                                 + 0.0548548661664975 w_k1_AXP_s000
                                 + 0.0259053612130968 w_k1_AXP_s001
                                 + 0.0243831856024519 w_k1_AXP_s002
                                 + 0.0259631850150718 w_k1_AXP_s003
                                 + 0.0605566890875953 w_k1_AXP_s004
                                 + 0.0209979022467801 w_k1_AXP_s005
                                 + 0.0259168032202017 w_k1_AXP_s006
                                 + 0.0446113942108861 w_k1_AXP_s007
                                 + 0.0109625548964595 w_k1_AXP_s010
                                 + 0.0368344244383317 w_k1_AXP_s011
                                 + 0.0281162293886101 w_k1_AXP_s012
                                 + 0.0516050858729154 w_k1_AXP_s013
                                 + 0.0135808269706773 w_k1_AXP_s014
                                 + 0.015132936146254 w_k1_AXP_s015
                                 + 0.0273122817633463 w_k1_AXP_s016
                                 + 0.0436404377437764 w_k1_AXP_s017
                                 + 0.0171932440112649 w_k1_AXP_s100
                                 + 0.0157026029805125 w_k1_AXP_s101
                                 + 0.0248001976919365 w_k1_AXP_s102
                                 + 0.0105707841929847 w_k1_AXP_s103
                                 + 0.0178558491795607 w_k1_AXP_s104
                                 + 0.0166581280755759 w_k1_AXP_s105
                                 + 0.0149203368869235 w_k1_AXP_s106
                                 + 0.0237382701166106 w_k1_AXP_s107
                                 + 0.0650618813629371 w_k1_AXP_s110
                                 + 0.0676032415220908 w_k1_AXP_s111
                                 + 0.0389651606092615 w_k1_AXP_s112
                                 + 0.036286785367948 w_k1_AXP_s113
                                 + 0.0543066685419865 w_k1_AXP_s114
                                 + 0.0534879235118442 w_k1_AXP_s115
                                 + 0.0258586502635249 w_k1_AXP_s116
                                 + 0.0276949862039869 w_k1_AXP_s117
                                 + 0.057095823224845 w_k1_DIS_s000
                                 + 0.027253743988135 w_k1_DIS_s001
                                 + 0.0225737459263967 w_k1_DIS_s002
                                 + 0.0278774786978097 w_k1_DIS_s003
                                 + 0.0620587041965682 w_k1_DIS_s004
                                 + 0.0228950386574439 w_k1_DIS_s005
                                 + 0.0254246031383081 w_k1_DIS_s006
                                 + 0.0495535582910408 w_k1_DIS_s007
                                 + 0.0111342408838118 w_k1_DIS_s010
                                 + 0.0390116633285311 w_k1_DIS_s011
                                 + 0.028015817480636 w_k1_DIS_s012
                                 + 0.0499144366866887 w_k1_DIS_s013
                                 + 0.0126256553836591 w_k1_DIS_s014
                                 + 0.0160727821602108 w_k1_DIS_s015
                                 + 0.0320629090722546 w_k1_DIS_s016
                                 + 0.0437385858204936 w_k1_DIS_s017
                                 + 0.0179704414147446 w_k1_DIS_s100
                                 + 0.0169282901668208 w_k1_DIS_s101
                                 + 0.0252792481984215 w_k1_DIS_s102
                                 + 0.0123242331137616 w_k1_DIS_s103
                                 + 0.0165774589256554 w_k1_DIS_s104
                                 + 0.0160511239999504 w_k1_DIS_s105
                                 + 0.0159078860357282 w_k1_DIS_s106
                                 + 0.0229308701358636 w_k1_DIS_s107
                                 + 0.0629091883561088 w_k1_DIS_s110
                                 + 0.0664227071425784 w_k1_DIS_s111
                                 + 0.0427531160823893 w_k1_DIS_s112
                                 + 0.0355852423240068 w_k1_DIS_s113
                                 + 0.0514180426082668 w_k1_DIS_s114
                                 + 0.0493672161052242 w_k1_DIS_s115
                                 + 0.025296981403086 w_k1_DIS_s116
                                 + 0.0277676865910258 w_k1_DIS_s117
                                 + 0.0558236188490741 w_k1_FDX_s000
                                 + 0.0267148330862307 w_k1_FDX_s001
                                 + 0.0225361720597992 w_k1_FDX_s002
                                 + 0.027231860797432 w_k1_FDX_s003
                                 + 0.0599620291609212 w_k1_FDX_s004
                                 + 0.0221068482681321 w_k1_FDX_s005
                                 + 0.0271585946930805 w_k1_FDX_s006
                                 + 0.0487417430193298 w_k1_FDX_s007
                                 + 0.0107207063984012 w_k1_FDX_s010
                                 + 0.0378793361754383 w_k1_FDX_s011
                                 + 0.0271824609203071 w_k1_FDX_s012
                                 + 0.0495982138142647 w_k1_FDX_s013
                                 + 0.0132353540811779 w_k1_FDX_s014
                                 + 0.0156424478875576 w_k1_FDX_s015
                                 + 0.0311167388288282 w_k1_FDX_s016
                                 + 0.0416895723613158 w_k1_FDX_s017
                                 + 0.017107843004095 w_k1_FDX_s100
                                 + 0.0166099012484633 w_k1_FDX_s101
                                 + 0.0238120036826276 w_k1_FDX_s102
                                 + 0.0114766165608611 w_k1_FDX_s103
                                 + 0.0181986229546137 w_k1_FDX_s104
                                 + 0.0156300641692066 w_k1_FDX_s105
                                 + 0.0152068027075908 w_k1_FDX_s106
                                 + 0.0218579989948474 w_k1_FDX_s107
                                 + 0.0656006823684151 w_k1_FDX_s110
                                 + 0.0699763770068066 w_k1_FDX_s111
                                 + 0.0400239120867466 w_k1_FDX_s112
                                 + 0.0368800113992482 w_k1_FDX_s113
                                 + 0.0518720836935468 w_k1_FDX_s114
                                 + 0.0503798685457025 w_k1_FDX_s115
                                 + 0.0262617617432643 w_k1_FDX_s116
                                 + 0.027132944442807 w_k1_FDX_s117
                                 + 0.058085971483265 w_k1_IBM_s000
                                 + 0.0265626402619211 w_k1_IBM_s001
                                 + 0.0237661221319949 w_k1_IBM_s002
                                 + 0.02792580479504 w_k1_IBM_s003
                                 + 0.0592909134083184 w_k1_IBM_s004
                                 + 0.0220037540616613 w_k1_IBM_s005
                                 + 0.0273402583638886 w_k1_IBM_s006
                                 + 0.0488615999799171 w_k1_IBM_s007
                                 + 0.0108548671644491 w_k1_IBM_s010
                                 + 0.0371769666591401 w_k1_IBM_s011
                                 + 0.0268296729806713 w_k1_IBM_s012
                                 + 0.049451510292155 w_k1_IBM_s013
                                 + 0.0129279402652954 w_k1_IBM_s014
                                 + 0.0159031593605726 w_k1_IBM_s015
                                 + 0.0310710910846739 w_k1_IBM_s016
                                 + 0.0424942234321868 w_k1_IBM_s017
                                 + 0.0183290052879958 w_k1_IBM_s100
                                 + 0.0165274362114167 w_k1_IBM_s101
                                 + 0.0249482567900509 w_k1_IBM_s102
                                 + 0.0123038717521316 w_k1_IBM_s103
                                 + 0.0170123633795872 w_k1_IBM_s104
                                 + 0.0158735130182076 w_k1_IBM_s105
                                 + 0.0153454288781703 w_k1_IBM_s106
                                 + 0.0227318763031534 w_k1_IBM_s107
                                 + 0.0607948542699377 w_k1_IBM_s110
                                 + 0.0732685326359115 w_k1_IBM_s111
                                 + 0.0403583285134576 w_k1_IBM_s112
                                 + 0.0356664812134075 w_k1_IBM_s113
                                 + 0.0527730727207091 w_k1_IBM_s114
                                 + 0.0512248102386278 w_k1_IBM_s115
                                 + 0.0274858892948353 w_k1_IBM_s116
                                 + 0.0276961223677052 w_k1_IBM_s117
                                 + 0.0591869108086535 w_k1_INTC_s000
                                 + 0.0238557928851295 w_k1_INTC_s001
                                 + 0.0236874192529417 w_k1_INTC_s002
                                 + 0.0317850967986033 w_k1_INTC_s003
                                 + 0.0703395455579377 w_k1_INTC_s004
                                 + 0.0228592181632977 w_k1_INTC_s005
                                 + 0.026343897973312 w_k1_INTC_s006
                                 + 0.0512604728184645 w_k1_INTC_s007
                                 + 0.00975798836718052 w_k1_INTC_s010
                                 + 0.0330147860653373 w_k1_INTC_s011
                                 + 0.0245285436389511 w_k1_INTC_s012
                                 + 0.0514123960094305 w_k1_INTC_s013
                                 + 0.0136897227141136 w_k1_INTC_s014
                                 + 0.0139547179934265 w_k1_INTC_s015
                                 + 0.030029733677497 w_k1_INTC_s016
                                 + 0.0391941245870792 w_k1_INTC_s017
                                 + 0.0187561335924955 w_k1_INTC_s100
                                 + 0.014116600252778 w_k1_INTC_s101
                                 + 0.027847629545514 w_k1_INTC_s102
                                 + 0.0118159220517737 w_k1_INTC_s103
                                 + 0.0194994059726987 w_k1_INTC_s104
                                 + 0.0134712345976066 w_k1_INTC_s105
                                 + 0.0143836597167808 w_k1_INTC_s106
                                 + 0.0222920659382311 w_k1_INTC_s107
                                 + 0.0643712626878216 w_k1_INTC_s110
                                 + 0.067426821856443 w_k1_INTC_s111
                                 + 0.0445978706810737 w_k1_INTC_s112
                                 + 0.034266073957382 w_k1_INTC_s113
                                 + 0.0547686050442841 w_k1_INTC_s114
                                 + 0.0562265559498598 w_k1_INTC_s115
                                 + 0.028487969055128 w_k1_INTC_s116
                                 + 0.0254601676158696 w_k1_INTC_s117
                                 + 0.054137305071816 w_k1_KO_s000
                                 + 0.0256060625182636 w_k1_KO_s001
                                 + 0.0240997087438075 w_k1_KO_s002
                                 + 0.0259717783433202 w_k1_KO_s003
                                 + 0.0579822203374912 w_k1_KO_s004
                                 + 0.0232831252026789 w_k1_KO_s005
                                 + 0.0244680778675702 w_k1_KO_s006
                                 + 0.0476265581253994 w_k1_KO_s007
                                 + 0.0106195023602444 w_k1_KO_s010
                                 + 0.0374577751848069 w_k1_KO_s011
                                 + 0.02792434026353 w_k1_KO_s012
                                 + 0.0456100979045602 w_k1_KO_s013
                                 + 0.012937876043214 w_k1_KO_s014
                                 + 0.0147632909033408 w_k1_KO_s015
                                 + 0.0314409343791339 w_k1_KO_s016
                                 + 0.0433282420527821 w_k1_KO_s017
                                 + 0.016697668899505 w_k1_KO_s100
                                 + 0.0163160362114916 w_k1_KO_s101
                                 + 0.0254634638112955 w_k1_KO_s102
                                 + 0.0112041494183787 w_k1_KO_s103
                                 + 0.0156348098822644 w_k1_KO_s104
                                 + 0.0155336535949184 w_k1_KO_s105
                                 + 0.0134094763333606 w_k1_KO_s106
                                 + 0.0221040239533986 w_k1_KO_s107
                                 + 0.062078878379167 w_k1_KO_s110
                                 + 0.0647027656496058 w_k1_KO_s111
                                 + 0.0406760758807877 w_k1_KO_s112
                                 + 0.0352899930618549 w_k1_KO_s113
                                 + 0.0508448449697128 w_k1_KO_s114
                                 + 0.0493740911674698 w_k1_KO_s115
                                 + 0.0243996073112984 w_k1_KO_s116
                                 + 0.0287033693905252 w_k1_KO_s117
                                 + 0.0491404302151315 w_k1_MAT_s000
                                 + 0.0239734264621634 w_k1_MAT_s001
                                 + 0.021376254728834 w_k1_MAT_s002
                                 + 0.0233288779411414 w_k1_MAT_s003
                                 + 0.0532583715621531 w_k1_MAT_s004
                                 + 0.0219013135352306 w_k1_MAT_s005
                                 + 0.0256706262767957 w_k1_MAT_s006
                                 + 0.0389967335351886 w_k1_MAT_s007
                                 + 0.0129720049954128 w_k1_MAT_s010
                                 + 0.0481118608795594 w_k1_MAT_s011
                                 + 0.0279267388265664 w_k1_MAT_s012
                                 + 0.0487468540559465 w_k1_MAT_s013
                                 + 0.0130107010041466 w_k1_MAT_s014
                                 + 0.0166265440762813 w_k1_MAT_s015
                                 + 0.0357867559247601 w_k1_MAT_s016
                                 + 0.0434931582634754 w_k1_MAT_s017
                                 + 0.0168421621862858 w_k1_MAT_s100
                                 + 0.017694053689174 w_k1_MAT_s101
                                 + 0.0230582586910612 w_k1_MAT_s102
                                 + 0.0118423035365031 w_k1_MAT_s103
                                 + 0.0162335788706907 w_k1_MAT_s104
                                 + 0.0165558084629888 w_k1_MAT_s105
                                 + 0.0172430143926749 w_k1_MAT_s106
                                 + 0.0237520723661887 w_k1_MAT_s107
                                 + 0.0570309437046838 w_k1_MAT_s110
                                 + 0.069113899267655 w_k1_MAT_s111
                                 + 0.0406697928997176 w_k1_MAT_s112
                                 + 0.0441354921587309 w_k1_MAT_s113
                                 + 0.0531175894923745 w_k1_MAT_s114
                                 + 0.0414517805419672 w_k1_MAT_s115
                                 + 0.0238205538578143 w_k1_MAT_s116
                                 + 0.0336175276475264 w_k1_MAT_s117
                                 + 0.0580176953172443 w_k1_MSFT_s000
                                 + 0.0244744918391026 w_k1_MSFT_s001
                                 + 0.021165485815513 w_k1_MSFT_s002
                                 + 0.0282697902264539 w_k1_MSFT_s003
                                 + 0.0613365484604609 w_k1_MSFT_s004
                                 + 0.0227656739907031 w_k1_MSFT_s005
                                 + 0.0248130576858831 w_k1_MSFT_s006
                                 + 0.049738645382288 w_k1_MSFT_s007
                                 + 0.00986781463083955 w_k1_MSFT_s010
                                 + 0.0354447192652488 w_k1_MSFT_s011
                                 + 0.0243802961534349 w_k1_MSFT_s012
                                 + 0.0524808863636839 w_k1_MSFT_s013
                                 + 0.0120640871288195 w_k1_MSFT_s014
                                 + 0.0143600277769669 w_k1_MSFT_s015
                                 + 0.0320974402874201 w_k1_MSFT_s016
                                 + 0.0406242285188061 w_k1_MSFT_s017
                                 + 0.0174688397691755 w_k1_MSFT_s100
                                 + 0.0156001172490324 w_k1_MSFT_s101
                                 + 0.0249049595673003 w_k1_MSFT_s102
                                 + 0.0114651595294812 w_k1_MSFT_s103
                                 + 0.01701286310857 w_k1_MSFT_s104
                                 + 0.0160254485285415 w_k1_MSFT_s105
                                 + 0.0157141141091913 w_k1_MSFT_s106
                                 + 0.0201856223475478 w_k1_MSFT_s107
                                 + 0.059530041593556 w_k1_MSFT_s110
                                 + 0.0665767928143494 w_k1_MSFT_s111
                                 + 0.0425405890117107 w_k1_MSFT_s112
                                 + 0.0368167973821163 w_k1_MSFT_s113
                                 + 0.0509430741183088 w_k1_MSFT_s114
                                 + 0.0507663017202618 w_k1_MSFT_s115
                                 + 0.024982375879465 w_k1_MSFT_s116
                                 + 0.0270898214250598 w_k1_MSFT_s117
                                 + 0.0503611301395683 w_k1_PG_s000
                                 + 0.0244791260921347 w_k1_PG_s001
                                 + 0.0223382888465893 w_k1_PG_s002
                                 + 0.0280731413992215 w_k1_PG_s003
                                 + 0.0607127092394596 w_k1_PG_s004
                                 + 0.0221223266531261 w_k1_PG_s005
                                 + 0.0257573728602259 w_k1_PG_s006
                                 + 0.0463219258919735 w_k1_PG_s007
                                 + 0.00982456967158032 w_k1_PG_s010
                                 + 0.0368679001074413 w_k1_PG_s011
                                 + 0.0263916117466324 w_k1_PG_s012
                                 + 0.0482900018885786 w_k1_PG_s013
                                 + 0.0134298712682028 w_k1_PG_s014
                                 + 0.0144920666281535 w_k1_PG_s015
                                 + 0.030980655262533 w_k1_PG_s016
                                 + 0.0400122355569075 w_k1_PG_s017
                                 + 0.0171720459868468 w_k1_PG_s100
                                 + 0.0162344512100245 w_k1_PG_s101
                                 + 0.0229238701949912 w_k1_PG_s102
                                 + 0.0111381386154964 w_k1_PG_s103
                                 + 0.0167382126568025 w_k1_PG_s104
                                 + 0.0150727225241692 w_k1_PG_s105
                                 + 0.0144120475084734 w_k1_PG_s106
                                 + 0.0225228361900157 w_k1_PG_s107
                                 + 0.0676771298975135 w_k1_PG_s110
                                 + 0.0716178684862447 w_k1_PG_s111
                                 + 0.037261423740782 w_k1_PG_s112
                                 + 0.0376083142648029 w_k1_PG_s113
                                 + 0.0581952319705116 w_k1_PG_s114
                                 + 0.0519618630908412 w_k1_PG_s115
                                 + 0.0245670019047416 w_k1_PG_s116
                                 + 0.0267051674460375 w_k1_PG_s117 >= 1.219
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
