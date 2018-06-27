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
 w_k1_ADM_s0_return_constraint:  1.15847 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.899571 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.884691 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.14104 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.00656 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.10563 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.10542 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.4324 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.20615 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.906995 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.989621 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.838983 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.98607 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.2655 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.840154 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.32403 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.953298 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.794076 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.99435 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.95547 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.66089 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.730181 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.05819 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.22927 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05307 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.719603 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.724467 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.880621 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.952009 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.736658 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.2956 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.43918 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15634 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.18375 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.989894 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.08362 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.06791 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.18446 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20256 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.870547 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.843071 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.28102 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.03889 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.29872 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.33106 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.57274 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.04444 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.826609 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.01452 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.636656 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.76454 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.4348 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.820319 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.29217 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.996953 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.885023 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.951775 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.1058 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.612038 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.669396 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.959524 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.48205 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.01957 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.604246 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.536077 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.912399 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.929387 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.64903 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.45145 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.75532 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.01733 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.10556 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.74383 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.16697 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.907074 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.11076 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29679 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.867593 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.884301 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.18892 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.932156 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.49963 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.49512 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.79778 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.09723 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.708222 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.868572 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.465346 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.817516 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.05834 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.723391 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.56589 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.886 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.635797 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.11912 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.1069 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.616947 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.697065 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.831305 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.43693 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.19012 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.735837 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.495733 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.90985 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.13476 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.625816 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.5006 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.51386 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.873476 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.04486 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.602941 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.1779 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.838124 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.25081 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18084 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.895021 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.944373 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.09399 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.940809 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.30041 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.16407 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.61214 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.13385 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.737411 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.880221 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.636853 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.908886 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1542 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.951876 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.38556 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.970849 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.868124 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.07351 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.907775 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.717527 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.742415 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.01912 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.20871 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.12425 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.775724 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.678405 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.852895 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.964771 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.748149 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.33459 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.25172 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.11319 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.11539 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.796082 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.00004 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.912743 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12762 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.0633 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.01402 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.810992 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.37309 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.985567 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.08834 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.39132 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.10478 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.727016 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.957299 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.25586 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.799887 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.627715 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.11599 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.568992 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.10862 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.03729 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.849322 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.939292 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.65248 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.732144 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.06122 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.65296 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.62971 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.939972 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.722876 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.495862 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.42609 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.2094 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.0118 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.09485 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.95985 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.601975 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.822993 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.643578 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.35312 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.955757 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.943919 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09789 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.978953 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.958417 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.06296 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.00612 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.19578 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.04572 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.51671 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.12797 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.890031 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.891424 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.662006 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.947705 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.26609 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.933025 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.29809 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.0251 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.84472 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02454 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.1139 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.797506 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.766228 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.853844 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.15227 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.968362 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.793758 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.585099 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.764377 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.891703 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.707125 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.32115 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.19564 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.07231 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.0005 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.672337 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.96438 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.822017 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.03164 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.08424 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.926503 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.886657 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.31888 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.956366 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.25363 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.34956 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.36324 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.923734 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.930815 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.13454 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.740939 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.842535 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.12182 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.619337 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.2925 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.92492 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.730564 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.982416 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.26986 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.643538 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.891087 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.825846 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.35188 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.11538 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.841474 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.606903 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.17378 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.15021 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.855841 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.23196 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.56594 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.807552 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.922913 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.582393 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17803 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.863596 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.08583 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.06848 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.005 w_0_MAT - w_k1_MAT_s1 + 0.99 b_k1_MAT_s1
                                 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.975421 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.41926 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.965924 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.09676 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.3335 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.12925 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.539461 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.849524 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.03734 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.586809 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.43665 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.924324 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.545586 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.34989 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.25945 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.938725 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.04129 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.81605 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.656966 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.02802 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.729353 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.60339 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.964792 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.604337 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.450397 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.31282 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.12612 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.78609 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.04021 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.99284 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.738375 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.0258 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.649307 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.25003 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.88138 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.06724 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1672 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.920087 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.972466 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.06324 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.947548 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.2535 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.11079 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.40564 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.02276 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.765958 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.862329 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.63389 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.905835 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.01671 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.901412 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.28706 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.937228 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.796027 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.06032 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.960444 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.760609 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.802198 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.02553 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.3081 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.19395 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.834283 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.747673 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.912579 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.03007 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.778871 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.35905 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.22709 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.02057 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.04697 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.709881 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01068 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.916622 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.1105 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06861 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.94964 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.942921 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.06381 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.986488 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12312 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.2862 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.26291 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.0742 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.856391 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.04221 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.771584 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.858444 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.04786 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.951287 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.35557 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.05164 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.994228 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.13885 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.26271 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.827765 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.910937 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.890828 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.23255 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.03922 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.797979 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.690204 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.999947 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.00723 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.825552 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.17231 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.27585 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.908613 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.04368 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.830271 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.976809 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.949152 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.13049 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.173999459168933 z_k1_s0000
                                 - 0.19799957295039 z_k1_s0001
                                 - 0.410998685890502 z_k1_s0002
                                 - 0.593999629348924 z_k1_s0003
                                 - 0.268999277263761 z_k1_s0004
                                 - 0.624999042860116 z_k1_s0005
                                 - 0.286999210850228 z_k1_s0006
                                 - 0.476997633042878 z_k1_s0007
                                 - 0.455000680200511 z_k1_s0010
                                 - 0.20000051542262 z_k1_s0011
                                 - 0.697002133289483 z_k1_s0012
                                 - 0.240000563027037 z_k1_s0013
                                 - 0.337000040446517 z_k1_s0014
                                 - 0.211000278853353 z_k1_s0015
                                 - 0.187000416731024 z_k1_s0016
                                 - 0.447000448759199 z_k1_s0017
                                 - 0.423997529978644 z_k1_s0020
                                 - 0.275999482357158 z_k1_s0021
                                 - 0.47600012953534 z_k1_s0022
                                 - 0.619997804022096 z_k1_s0023
                                 - 0.615000151597167 z_k1_s0024
                                 - 0.440997802604285 z_k1_s0025
                                 - 0.671996651587482 z_k1_s0026
                                 - 0.227999507129978 z_k1_s0027
                                 - 0.184999522989574 z_k1_s0030
                                 - 0.67699973591057 z_k1_s0031
                                 - 0.310998568780036 z_k1_s0032
                                 - 0.17699961650122 z_k1_s0033
                                 - 0.22899952727441 z_k1_s0034
                                 - 0.342000410467692 z_k1_s0035
                                 - 0.178999593123309 z_k1_s0036
                                 - 0.4929982564862 z_k1_s0037
                                 - 0.786999140060023 z_k1_s0040
                                 - 0.352999952906998 z_k1_s0041
                                 - 0.495999520688867 z_k1_s0042
                                 - 0.600997090253211 z_k1_s0043
                                 - 0.444999517089846 z_k1_s0044
                                 - 0.356999500593503 z_k1_s0045
                                 - 0.931001742361383 z_k1_s0046
                                 - 1.27599945229751 z_k1_s0047
                                 - 0.158999999747435 z_k1_s0050
                                 - 0.421999996507028 z_k1_s0051
                                 - 0.306999460387702 z_k1_s0052
                                 - 0.586000526903236 z_k1_s0053
                                 - 0.267000151998555 z_k1_s0054
                                 - 0.168999939044745 z_k1_s0055
                                 - 0.200000132346487 z_k1_s0056
                                 - 0.134000039304136 z_k1_s0057
                                 - 0.606000538929473 z_k1_s0060
                                 - 0.541000123471149 z_k1_s0061
                                 - 0.32499910828712 z_k1_s0062
                                 - 0.454999939203767 z_k1_s0063
                                 - 0.144000280922086 z_k1_s0064
                                 - 0.314001915629301 z_k1_s0065
                                 - 0.37300197299868 z_k1_s0066
                                 - 0.211000320909309 z_k1_s0067
                                 - 0.153000519604184 z_k1_s0070
                                 - 0.430002797585976 z_k1_s0071
                                 - 0.36800055439147 z_k1_s0072
                                 - 0.208000775927547 z_k1_s0073
                                 - 0.287000439778161 z_k1_s0074
                                 - 0.374002179480695 z_k1_s0075
                                 - 0.556001460311454 z_k1_s0076
                                 - 0.352000767505876 z_k1_s0077
                                 - 0.196000253560562 z_k1_s0100
                                 - 0.171999811084401 z_k1_s0101
                                 - 0.133999530496491 z_k1_s0102
                                 - 0.0979998747806747 z_k1_s0103
                                 - 0.134999968932341 z_k1_s0104
                                 - 0.0919998901614377 z_k1_s0105
                                 - 0.114999768201945 z_k1_s0106
                                 - 0.317999562818972 z_k1_s0107
                                 - 0.645999304925116 z_k1_s0110
                                 - 0.212999692295294 z_k1_s0111
                                 - 0.66699919644454 z_k1_s0112
                                 - 0.246999384560361 z_k1_s0113
                                 - 0.336999315942746 z_k1_s0114
                                 - 0.969997631224794 z_k1_s0115
                                 - 0.475000188266953 z_k1_s0116
                                 - 0.407997892377372 z_k1_s0117
                                 - 0.558999328718311 z_k1_s0120
                                 - 0.181999748792063 z_k1_s0121
                                 - 0.182999629126282 z_k1_s0122
                                 - 0.146999725101401 z_k1_s0123
                                 - 0.247999806837288 z_k1_s0124
                                 - 0.325998896892365 z_k1_s0125
                                 - 0.521000132024212 z_k1_s0126
                                 - 0.173999770119867 z_k1_s0127
                                 - 0.137000174709172 z_k1_s0130
                                 - 0.0890001890883231 z_k1_s0131
                                 - 0.0939995403414978 z_k1_s0132
                                 - 0.141999525962347 z_k1_s0133
                                 - 0.101999537944973 z_k1_s0134
                                 - 0.138999742610709 z_k1_s0135
                                 - 0.0839998885366193 z_k1_s0136
                                 - 0.0759998909331445 z_k1_s0137
                                 - 0.832000198737844 z_k1_s0140
                                 - 0.451999210674617 z_k1_s0141
                                 - 0.275000171645593 z_k1_s0142
                                 - 0.605998607899939 z_k1_s0143
                                 - 0.341000196956531 z_k1_s0144
                                 - 0.220000216736498 z_k1_s0145
                                 - 0.995001855665335 z_k1_s0146
                                 - 0.249999939441383 z_k1_s0147
                                 - 0.253000081380732 z_k1_s0150
                                 - 0.177000340570414 z_k1_s0151
                                 - 0.0849999304323627 z_k1_s0152
                                 - 0.221999793682196 z_k1_s0153
                                 - 0.231999838262419 z_k1_s0154
                                 - 0.207000474311083 z_k1_s0155
                                 - 0.142999889997968 z_k1_s0156
                                 - 0.176000186612548 z_k1_s0157
                                 - 0.137999813500656 z_k1_s0160
                                 - 0.085999946036566 z_k1_s0161
                                 - 0.256999300905264 z_k1_s0162
                                 - 0.228999438362983 z_k1_s0163
                                 - 0.0919998756671037 z_k1_s0164
                                 - 0.142999778726075 z_k1_s0165
                                 - 0.392999042395855 z_k1_s0166
                                 - 0.0939999476771688 z_k1_s0167
                                 - 0.0979999176344813 z_k1_s0170
                                 - 0.211999552189132 z_k1_s0171
                                 - 0.287999308558899 z_k1_s0172
                                 - 0.396999673825392 z_k1_s0173
                                 - 0.114999928811867 z_k1_s0174
                                 - 0.115999929469361 z_k1_s0175
                                 - 0.108999924866908 z_k1_s0176
                                 - 0.132999940646747 z_k1_s0177
                                 - 0.798001617529151 z_k1_s1000
                                 - 1.51200612353327 z_k1_s1001
                                 - 0.729001361144134 z_k1_s1002
                                 - 1.65700574986281 z_k1_s1003
                                 - 0.911001560124225 z_k1_s1004
                                 - 1.8070033593043 z_k1_s1005
                                 - 0.435000816191361 z_k1_s1006
                                 - 1.83700094398984 z_k1_s1007
                                 - 1.24999608257072 z_k1_s1010
                                 - 2.40599827592859 z_k1_s1011
                                 - 1.09500063436595 z_k1_s1012
                                 - 1.06000069985795 z_k1_s1013
                                 - 1.35100609023311 z_k1_s1014
                                 - 2.05999867081507 z_k1_s1015
                                 - 0.721000639685818 z_k1_s1016
                                 - 1.10600544377656 z_k1_s1017
                                 - 1.33199842126714 z_k1_s1020
                                 - 0.728001272298155 z_k1_s1021
                                 - 0.534000449139198 z_k1_s1022
                                 - 0.790000703859166 z_k1_s1023
                                 - 2.56400030463288 z_k1_s1024
                                 - 0.743000752983365 z_k1_s1025
                                 - 0.635000809904851 z_k1_s1026
                                 - 1.44099868040514 z_k1_s1027
                                 - 0.29600031690735 z_k1_s1030
                                 - 0.260000294575387 z_k1_s1031
                                 - 0.198000122825797 z_k1_s1032
                                 - 0.540999002709916 z_k1_s1033
                                 - 0.189000117242807 z_k1_s1034
                                 - 0.342998879884119 z_k1_s1035
                                 - 0.625999721883094 z_k1_s1036
                                 - 0.546001338368839 z_k1_s1037
                                 - 0.283000205069451 z_k1_s1040
                                 - 0.287000058950221 z_k1_s1041
                                 - 0.385001345031379 z_k1_s1042
                                 - 0.193000248750595 z_k1_s1043
                                 - 0.639000176464102 z_k1_s1044
                                 - 0.23700026343983 z_k1_s1045
                                 - 0.479998685700053 z_k1_s1046
                                 - 0.739998919954694 z_k1_s1047
                                 - 0.279000192848756 z_k1_s1050
                                 - 0.611000403680599 z_k1_s1051
                                 - 0.302000252632009 z_k1_s1052
                                 - 0.579000693147569 z_k1_s1053
                                 - 0.59099981934706 z_k1_s1054
                                 - 0.195000187449158 z_k1_s1055
                                 - 0.242000030132183 z_k1_s1056
                                 - 0.262000002265406 z_k1_s1057
                                 - 0.372000497868981 z_k1_s1060
                                 - 0.20500004675031 z_k1_s1061
                                 - 0.281000145213864 z_k1_s1062
                                 - 0.275999959683312 z_k1_s1063
                                 - 0.718000635779279 z_k1_s1064
                                 - 0.214000078305211 z_k1_s1065
                                 - 0.679000095841252 z_k1_s1066
                                 - 0.279000071001643 z_k1_s1067
                                 - 0.553001091123488 z_k1_s1070
                                 - 0.462000525141184 z_k1_s1071
                                 - 0.553998339013707 z_k1_s1072
                                 - 0.186999921814321 z_k1_s1073
                                 - 0.420000785764679 z_k1_s1074
                                 - 0.36800143148897 z_k1_s1075
                                 - 0.451000270352135 z_k1_s1076
                                 - 0.396999636296735 z_k1_s1077
                                 - 0.101000184672446 z_k1_s1100
                                 - 0.249000107360354 z_k1_s1101
                                 - 0.26600032834506 z_k1_s1102
                                 - 0.0600000140627214 z_k1_s1103
                                 - 0.184000827126647 z_k1_s1104
                                 - 0.13800013034438 z_k1_s1105
                                 - 0.263000124641673 z_k1_s1106
                                 - 0.139000151578785 z_k1_s1107
                                 - 0.06099977031396 z_k1_s1110
                                 - 0.169999781201405 z_k1_s1111
                                 - 0.164999863224636 z_k1_s1112
                                 - 0.0559998523371907 z_k1_s1113
                                 - 0.148999097702474 z_k1_s1114
                                 - 0.263998496163792 z_k1_s1115
                                 - 0.196998824277709 z_k1_s1116
                                 - 0.222999425753409 z_k1_s1117
                                 - 0.0639999077873613 z_k1_s1120
                                 - 0.0759998189976393 z_k1_s1121
                                 - 0.0869997722736558 z_k1_s1122
                                 - 0.318999581002545 z_k1_s1123
                                 - 0.127999815574723 z_k1_s1124
                                 - 0.094999851746888 z_k1_s1125
                                 - 0.100999807352027 z_k1_s1126
                                 - 0.170000128809406 z_k1_s1127
                                 - 0.10999972358684 z_k1_s1130
                                 - 0.0899999068436431 z_k1_s1131
                                 - 0.232000215018708 z_k1_s1132
                                 - 0.100999879202327 z_k1_s1133
                                 - 0.0599999727290601 z_k1_s1134
                                 - 0.161999915918473 z_k1_s1135
                                 - 0.172999574777474 z_k1_s1136
                                 - 0.11700017149644 z_k1_s1137
                                 - 0.168999962329691 z_k1_s1140
                                 - 0.41099904309553 z_k1_s1141
                                 - 0.249000889174246 z_k1_s1142
                                 - 0.172999941591938 z_k1_s1143
                                 - 0.502999236927016 z_k1_s1144
                                 - 0.172999941591938 z_k1_s1145
                                 - 0.311000679539041 z_k1_s1146
                                 - 0.246999669743473 z_k1_s1147
                                 - 0.231999840487012 z_k1_s1150
                                 - 0.151999497782417 z_k1_s1151
                                 - 0.477997434234702 z_k1_s1152
                                 - 0.473998393016353 z_k1_s1153
                                 - 0.273999754247776 z_k1_s1154
                                 - 0.294999711128157 z_k1_s1155
                                 - 0.131999633905559 z_k1_s1156
                                 - 0.180999311493825 z_k1_s1157
                                 - 0.353000467140827 z_k1_s1160
                                 - 0.502999167973935 z_k1_s1161
                                 - 0.141999842800055 z_k1_s1162
                                 - 0.535999341876381 z_k1_s1163
                                 - 0.237999675372034 z_k1_s1164
                                 - 0.263998837224222 z_k1_s1165
                                 - 0.314999570345339 z_k1_s1166
                                 - 0.202999723111441 z_k1_s1167
                                 - 0.0499999384092772 z_k1_s1170
                                 - 0.0829999272373776 z_k1_s1171
                                 - 0.131000131678082 z_k1_s1172
                                 - 0.201999574305615 z_k1_s1173
                                 - 0.111999841228797 z_k1_s1174
                                 - 0.089000082842366 z_k1_s1175
                                 - 0.10600020582341 z_k1_s1176
                                 - 0.0940000680133004 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.22104 w_k1_ADM_s000
                                 + 0.782755 w_k1_AXP_s000
                                 + 0.957462 w_k1_DIS_s000
                                 + 1.07661 w_k1_FDX_s000
                                 + 0.67309 w_k1_IBM_s000
                                 + 1.12664 w_k1_INTC_s000
                                 + 0.917259 w_k1_KO_s000
                                 + 0.567309 w_k1_MAT_s000
                                 + 1.1044 w_k1_MSFT_s000
                                 + 0.884811 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.35578 w_k1_ADM_s000
                                 + 1.6448 w_k1_AXP_s000 + 1.64363 w_k1_DIS_s000
                                 + 1.43707 w_k1_FDX_s000
                                 + 1.68694 w_k1_IBM_s000
                                 + 1.51566 w_k1_INTC_s000
                                 + 1.61342 w_k1_KO_s000 + 1.71557 w_k1_MAT_s000
                                 + 1.44029 w_k1_MSFT_s000
                                 + 1.16379 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.53304 w_k1_ADM_s000
                                 + 1.18669 w_k1_AXP_s000
                                 + 1.33904 w_k1_DIS_s000
                                 + 1.35079 w_k1_FDX_s000
                                 + 1.00967 w_k1_IBM_s000
                                 + 1.12402 w_k1_INTC_s000
                                 + 1.05453 w_k1_KO_s000
                                 + 0.813382 w_k1_MAT_s000
                                 + 1.19598 w_k1_MSFT_s000
                                 + 0.976816 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.979536 w_k1_ADM_s000
                                 + 0.957147 w_k1_AXP_s000
                                 + 1.04962 w_k1_DIS_s000
                                 + 1.02136 w_k1_FDX_s000
                                 + 1.18384 w_k1_IBM_s000
                                 + 1.10327 w_k1_INTC_s000
                                 + 1.23431 w_k1_KO_s000 + 1.1624 w_k1_MAT_s000
                                 + 1.18186 w_k1_MSFT_s000
                                 + 0.932977 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.22425 w_k1_ADM_s000
                                 + 1.20688 w_k1_AXP_s000 + 1.0856 w_k1_DIS_s000
                                 + 1.14055 w_k1_FDX_s000
                                 + 0.957511 w_k1_IBM_s000
                                 + 1.01628 w_k1_INTC_s000
                                 + 1.03585 w_k1_KO_s000 + 1.10144 w_k1_MAT_s000
                                 + 1.06031 w_k1_MSFT_s000
                                 + 1.01559 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.981781 w_k1_ADM_s000
                                 + 1.17188 w_k1_AXP_s000
                                 + 0.943863 w_k1_DIS_s000
                                 + 0.999094 w_k1_FDX_s000
                                 + 1.20177 w_k1_IBM_s000
                                 + 1.17955 w_k1_INTC_s000
                                 + 1.04241 w_k1_KO_s000 + 1.07734 w_k1_MAT_s000
                                 + 0.971241 w_k1_MSFT_s000
                                 + 0.923845 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.923489 w_k1_ADM_s000
                                 + 0.728672 w_k1_AXP_s000
                                 + 0.61061 w_k1_DIS_s000
                                 + 0.71973 w_k1_FDX_s000
                                 + 0.832472 w_k1_IBM_s000
                                 + 0.848799 w_k1_INTC_s000
                                 + 0.762919 w_k1_KO_s000
                                 + 0.594874 w_k1_MAT_s000
                                 + 0.779613 w_k1_MSFT_s000
                                 + 0.778559 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.1268 w_k1_ADM_s000
                                 + 0.9164 w_k1_AXP_s000
                                 + 0.739531 w_k1_DIS_s000
                                 + 0.885725 w_k1_FDX_s000
                                 + 0.752221 w_k1_IBM_s000
                                 + 1.14008 w_k1_INTC_s000
                                 + 0.794952 w_k1_KO_s000
                                 + 0.723619 w_k1_MAT_s000
                                 + 0.928379 w_k1_MSFT_s000
                                 + 0.833627 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.807096 w_k1_ADM_s001
                                 + 0.932614 w_k1_AXP_s001
                                 + 0.768417 w_k1_DIS_s001
                                 + 0.919348 w_k1_FDX_s001
                                 + 0.645075 w_k1_IBM_s001
                                 + 0.768088 w_k1_INTC_s001
                                 + 0.825882 w_k1_KO_s001
                                 + 0.92188 w_k1_MAT_s001
                                 + 0.821245 w_k1_MSFT_s001
                                 + 0.949467 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.50457 w_k1_ADM_s001
                                 + 1.2938 w_k1_AXP_s001 + 1.65528 w_k1_DIS_s001
                                 + 1.47624 w_k1_FDX_s001 + 1.026 w_k1_IBM_s001
                                 + 1.15661 w_k1_INTC_s001
                                 + 1.38797 w_k1_KO_s001 + 1.03266 w_k1_MAT_s001
                                 + 1.21238 w_k1_MSFT_s001
                                 + 1.21714 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.02318 w_k1_ADM_s001
                                 + 0.936832 w_k1_AXP_s001
                                 + 0.847024 w_k1_DIS_s001
                                 + 0.902086 w_k1_FDX_s001
                                 + 1.07151 w_k1_IBM_s001
                                 + 0.878263 w_k1_INTC_s001
                                 + 1.01375 w_k1_KO_s001 + 1.0271 w_k1_MAT_s001
                                 + 0.874045 w_k1_MSFT_s001
                                 + 0.998275 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.883337 w_k1_ADM_s001
                                 + 0.857352 w_k1_AXP_s001
                                 + 1.19582 w_k1_DIS_s001
                                 + 0.979612 w_k1_FDX_s001
                                 + 0.976212 w_k1_IBM_s001
                                 + 0.948708 w_k1_INTC_s001
                                 + 1.04677 w_k1_KO_s001
                                 + 0.770865 w_k1_MAT_s001
                                 + 1.04476 w_k1_MSFT_s001
                                 + 0.919393 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.07723 w_k1_ADM_s001
                                 + 1.28336 w_k1_AXP_s001
                                 + 1.12648 w_k1_DIS_s001
                                 + 1.14965 w_k1_FDX_s001
                                 + 0.876264 w_k1_IBM_s001
                                 + 1.08576 w_k1_INTC_s001
                                 + 1.13784 w_k1_KO_s001
                                 + 0.886038 w_k1_MAT_s001
                                 + 1.10636 w_k1_MSFT_s001
                                 + 1.07577 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.803725 w_k1_ADM_s001
                                 + 0.605369 w_k1_AXP_s001
                                 + 0.570036 w_k1_DIS_s001
                                 + 0.725898 w_k1_FDX_s001
                                 + 0.625601 w_k1_IBM_s001
                                 + 0.886447 w_k1_INTC_s001
                                 + 0.727606 w_k1_KO_s001
                                 + 0.456792 w_k1_MAT_s001
                                 + 0.885156 w_k1_MSFT_s001
                                 + 0.804892 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.803732 w_k1_ADM_s001
                                 + 0.618211 w_k1_AXP_s001
                                 + 0.62562 w_k1_DIS_s001
                                 + 0.657493 w_k1_FDX_s001
                                 + 0.900675 w_k1_IBM_s001
                                 + 0.855538 w_k1_INTC_s001
                                 + 0.86829 w_k1_KO_s001
                                 + 0.497501 w_k1_MAT_s001
                                 + 0.792141 w_k1_MSFT_s001
                                 + 0.869982 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.12404 w_k1_ADM_s001
                                 + 1.34067 w_k1_AXP_s001
                                 + 1.50589 w_k1_DIS_s001
                                 + 1.11789 w_k1_FDX_s001
                                 + 1.49853 w_k1_IBM_s001
                                 + 1.09797 w_k1_INTC_s001
                                 + 1.33391 w_k1_KO_s001 + 1.4244 w_k1_MAT_s001
                                 + 1.03659 w_k1_MSFT_s001
                                 + 1.16372 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08223 w_k1_ADM_s002
                                 + 0.962763 w_k1_AXP_s002
                                 + 0.884814 w_k1_DIS_s002
                                 + 1.06907 w_k1_FDX_s002
                                 + 0.627577 w_k1_IBM_s002
                                 + 0.904254 w_k1_INTC_s002
                                 + 0.70375 w_k1_KO_s002
                                 + 0.719467 w_k1_MAT_s002
                                 + 0.984014 w_k1_MSFT_s002
                                 + 0.841939 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.01932 w_k1_ADM_s002
                                 + 1.12401 w_k1_AXP_s002
                                 + 0.870405 w_k1_DIS_s002
                                 + 0.905491 w_k1_FDX_s002
                                 + 1.11059 w_k1_IBM_s002
                                 + 0.728867 w_k1_INTC_s002
                                 + 0.965161 w_k1_KO_s002
                                 + 1.06658 w_k1_MAT_s002
                                 + 0.834452 w_k1_MSFT_s002
                                 + 0.945227 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.878579 w_k1_ADM_s002
                                 + 1.16041 w_k1_AXP_s002
                                 + 1.06675 w_k1_DIS_s002 + 1.012 w_k1_FDX_s002
                                 + 1.43942 w_k1_IBM_s002
                                 + 1.07409 w_k1_INTC_s002 + 1.2338 w_k1_KO_s002
                                 + 2.1106 w_k1_MAT_s002
                                 + 0.987004 w_k1_MSFT_s002
                                 + 1.03566 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.830629 w_k1_ADM_s002
                                 + 0.810066 w_k1_AXP_s002
                                 + 0.619902 w_k1_DIS_s002
                                 + 0.739498 w_k1_FDX_s002
                                 + 0.955647 w_k1_IBM_s002
                                 + 0.758898 w_k1_INTC_s002
                                 + 0.911792 w_k1_KO_s002
                                 + 1.06833 w_k1_MAT_s002
                                 + 0.800712 w_k1_MSFT_s002
                                 + 0.907971 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.855598 w_k1_ADM_s002
                                 + 0.771308 w_k1_AXP_s002
                                 + 0.854168 w_k1_DIS_s002
                                 + 0.90661 w_k1_FDX_s002
                                 + 0.943629 w_k1_IBM_s002
                                 + 0.99456 w_k1_INTC_s002
                                 + 1.06863 w_k1_KO_s002 + 1.11995 w_k1_MAT_s002
                                 + 1.08638 w_k1_MSFT_s002
                                 + 0.921939 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.1568 w_k1_ADM_s002
                                 + 1.28822 w_k1_AXP_s002
                                 + 1.21332 w_k1_DIS_s002
                                 + 1.22909 w_k1_FDX_s002
                                 + 0.904702 w_k1_IBM_s002
                                 + 1.06674 w_k1_INTC_s002
                                 + 0.971079 w_k1_KO_s002
                                 + 0.983768 w_k1_MAT_s002
                                 + 1.12712 w_k1_MSFT_s002
                                 + 0.983846 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.14811 w_k1_ADM_s002
                                 + 1.39185 w_k1_AXP_s002
                                 + 1.51548 w_k1_DIS_s002
                                 + 1.24499 w_k1_FDX_s002
                                 + 1.41373 w_k1_IBM_s002
                                 + 1.00882 w_k1_INTC_s002 + 1.3259 w_k1_KO_s002
                                 + 1.59517 w_k1_MAT_s002
                                 + 1.20225 w_k1_MSFT_s002
                                 + 1.18631 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.18773 w_k1_ADM_s002
                                 + 0.872937 w_k1_AXP_s002
                                 + 1.17226 w_k1_DIS_s002
                                 + 1.05993 w_k1_FDX_s002
                                 + 1.06643 w_k1_IBM_s002
                                 + 1.00774 w_k1_INTC_s002
                                 + 1.20976 w_k1_KO_s002 + 1.18392 w_k1_MAT_s002
                                 + 1.16803 w_k1_MSFT_s002
                                 + 0.973733 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.743988 w_k1_ADM_s003
                                 + 0.706571 w_k1_AXP_s003
                                 + 0.85835 w_k1_DIS_s003
                                 + 0.968627 w_k1_FDX_s003
                                 + 0.828673 w_k1_IBM_s003
                                 + 1.03129 w_k1_INTC_s003
                                 + 0.889657 w_k1_KO_s003
                                 + 0.790337 w_k1_MAT_s003
                                 + 1.04976 w_k1_MSFT_s003
                                 + 0.974916 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.796294 w_k1_ADM_s003
                                 + 0.832812 w_k1_AXP_s003
                                 + 0.912935 w_k1_DIS_s003
                                 + 0.842541 w_k1_FDX_s003
                                 + 1.15066 w_k1_IBM_s003
                                 + 0.875525 w_k1_INTC_s003
                                 + 1.09662 w_k1_KO_s003 + 1.11201 w_k1_MAT_s003
                                 + 0.971615 w_k1_MSFT_s003
                                 + 1.04646 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.05411 w_k1_ADM_s003
                                 + 1.20101 w_k1_AXP_s003
                                 + 1.51447 w_k1_DIS_s003
                                 + 1.37783 w_k1_FDX_s003
                                 + 1.03999 w_k1_IBM_s003
                                 + 1.22791 w_k1_INTC_s003
                                 + 1.05812 w_k1_KO_s003
                                 + 0.818658 w_k1_MAT_s003
                                 + 1.16584 w_k1_MSFT_s003
                                 + 1.31058 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.80613 w_k1_ADM_s003
                                 + 0.659538 w_k1_AXP_s003
                                 + 1.15781 w_k1_DIS_s003
                                 + 1.01757 w_k1_FDX_s003
                                 + 0.785031 w_k1_IBM_s003
                                 + 1.00461 w_k1_INTC_s003
                                 + 0.969602 w_k1_KO_s003
                                 + 0.466512 w_k1_MAT_s003
                                 + 1.06155 w_k1_MSFT_s003
                                 + 1.07765 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.642449 w_k1_ADM_s003
                                 + 0.662207 w_k1_AXP_s003
                                 + 0.654949 w_k1_DIS_s003
                                 + 0.86702 w_k1_FDX_s003
                                 + 0.60176 w_k1_IBM_s003
                                 + 0.890082 w_k1_INTC_s003
                                 + 0.603679 w_k1_KO_s003
                                 + 0.491402 w_k1_MAT_s003
                                 + 0.893087 w_k1_MSFT_s003
                                 + 0.862513 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.807037 w_k1_ADM_s003
                                 + 1.27124 w_k1_AXP_s003
                                 + 1.26742 w_k1_DIS_s003
                                 + 1.10431 w_k1_FDX_s003
                                 + 1.60223 w_k1_IBM_s003
                                 + 0.972047 w_k1_INTC_s003
                                 + 1.31756 w_k1_KO_s003 + 1.61025 w_k1_MAT_s003
                                 + 1.19713 w_k1_MSFT_s003
                                 + 1.23517 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.661389 w_k1_ADM_s003
                                 + 0.741656 w_k1_AXP_s003
                                 + 0.646547 w_k1_DIS_s003
                                 + 0.77289 w_k1_FDX_s003
                                 + 0.783603 w_k1_IBM_s003
                                 + 0.781132 w_k1_INTC_s003
                                 + 0.716993 w_k1_KO_s003
                                 + 0.720383 w_k1_MAT_s003
                                 + 0.823838 w_k1_MSFT_s003
                                 + 0.844904 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.15 w_k1_ADM_s003
                                 + 1.03853 w_k1_AXP_s003
                                 + 0.979513 w_k1_DIS_s003
                                 + 1.13626 w_k1_FDX_s003
                                 + 0.785247 w_k1_IBM_s003
                                 + 0.89768 w_k1_INTC_s003
                                 + 0.856386 w_k1_KO_s003
                                 + 0.827708 w_k1_MAT_s003
                                 + 0.98983 w_k1_MSFT_s003
                                 + 1.16168 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.19433 w_k1_ADM_s004
                                 + 1.31353 w_k1_AXP_s004
                                 + 1.14808 w_k1_DIS_s004
                                 + 1.18531 w_k1_FDX_s004
                                 + 0.775905 w_k1_IBM_s004
                                 + 1.21514 w_k1_INTC_s004
                                 + 0.927453 w_k1_KO_s004
                                 + 0.755492 w_k1_MAT_s004
                                 + 1.08743 w_k1_MSFT_s004
                                 + 0.994425 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.13979 w_k1_ADM_s004
                                 + 1.47217 w_k1_AXP_s004
                                 + 1.76132 w_k1_DIS_s004
                                 + 1.40558 w_k1_FDX_s004 + 1.2182 w_k1_IBM_s004
                                 + 1.39349 w_k1_INTC_s004
                                 + 1.22258 w_k1_KO_s004 + 1.19016 w_k1_MAT_s004
                                 + 1.34202 w_k1_MSFT_s004
                                 + 1.14858 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.10178 w_k1_ADM_s004
                                 + 1.07314 w_k1_AXP_s004
                                 + 0.963306 w_k1_DIS_s004
                                 + 0.966239 w_k1_FDX_s004
                                 + 1.30573 w_k1_IBM_s004
                                 + 1.1719 w_k1_INTC_s004 + 1.07146 w_k1_KO_s004
                                 + 1.07028 w_k1_MAT_s004
                                 + 1.04925 w_k1_MSFT_s004
                                 + 0.987628 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.934541 w_k1_ADM_s004
                                 + 0.870344 w_k1_AXP_s004
                                 + 0.694486 w_k1_DIS_s004
                                 + 0.799235 w_k1_FDX_s004
                                 + 1.00435 w_k1_IBM_s004
                                 + 0.799623 w_k1_INTC_s004
                                 + 0.944822 w_k1_KO_s004
                                 + 1.14894 w_k1_MAT_s004
                                 + 0.836121 w_k1_MSFT_s004
                                 + 0.957209 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.18095 w_k1_ADM_s004
                                 + 1.09489 w_k1_AXP_s004 + 1.2219 w_k1_DIS_s004
                                 + 1.11371 w_k1_FDX_s004
                                 + 1.64297 w_k1_IBM_s004
                                 + 1.21196 w_k1_INTC_s004
                                 + 1.32491 w_k1_KO_s004 + 1.48065 w_k1_MAT_s004
                                 + 1.146 w_k1_MSFT_s004 + 1.07199 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.881671 w_k1_ADM_s004
                                 + 0.60719 w_k1_AXP_s004
                                 + 0.623882 w_k1_DIS_s004
                                 + 0.717467 w_k1_FDX_s004
                                 + 0.803132 w_k1_IBM_s004
                                 + 0.828616 w_k1_INTC_s004
                                 + 0.689127 w_k1_KO_s004
                                 + 0.668706 w_k1_MAT_s004
                                 + 0.73632 w_k1_MSFT_s004
                                 + 0.776945 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.965471 w_k1_ADM_s004
                                 + 0.853977 w_k1_AXP_s004
                                 + 0.818352 w_k1_DIS_s004
                                 + 0.964355 w_k1_FDX_s004
                                 + 0.623092 w_k1_IBM_s004
                                 + 0.885087 w_k1_INTC_s004
                                 + 0.705442 w_k1_KO_s004
                                 + 0.54371 w_k1_MAT_s004
                                 + 0.847424 w_k1_MSFT_s004
                                 + 0.840202 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.02374 w_k1_ADM_s004
                                 + 1.08118 w_k1_AXP_s004
                                 + 1.12684 w_k1_DIS_s004
                                 + 1.09329 w_k1_FDX_s004
                                 + 0.97103 w_k1_IBM_s004
                                 + 1.18263 w_k1_INTC_s004
                                 + 0.939431 w_k1_KO_s004
                                 + 0.934681 w_k1_MAT_s004
                                 + 1.02189 w_k1_MSFT_s004
                                 + 1.06358 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.39208 w_k1_ADM_s005
                                 + 1.32962 w_k1_AXP_s005
                                 + 1.33508 w_k1_DIS_s005
                                 + 1.29789 w_k1_FDX_s005
                                 + 1.13206 w_k1_IBM_s005
                                 + 1.20015 w_k1_INTC_s005
                                 + 1.04984 w_k1_KO_s005 + 1.54332 w_k1_MAT_s005
                                 + 1.28524 w_k1_MSFT_s005
                                 + 1.18296 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.700926 w_k1_ADM_s005
                                 + 0.813547 w_k1_AXP_s005
                                 + 0.842654 w_k1_DIS_s005
                                 + 0.842064 w_k1_FDX_s005
                                 + 1.11734 w_k1_IBM_s005
                                 + 0.997467 w_k1_INTC_s005
                                 + 0.873833 w_k1_KO_s005
                                 + 1.11079 w_k1_MAT_s005
                                 + 0.88902 w_k1_MSFT_s005
                                 + 0.955877 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.19244 w_k1_ADM_s005
                                 + 1.15756 w_k1_AXP_s005
                                 + 1.68354 w_k1_DIS_s005 + 1.236 w_k1_FDX_s005
                                 + 1.63143 w_k1_IBM_s005
                                 + 1.26408 w_k1_INTC_s005
                                 + 1.41744 w_k1_KO_s005 + 1.4916 w_k1_MAT_s005
                                 + 1.06316 w_k1_MSFT_s005
                                 + 1.15697 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.742824 w_k1_ADM_s005
                                 + 0.79814 w_k1_AXP_s005
                                 + 0.724171 w_k1_DIS_s005
                                 + 0.923173 w_k1_FDX_s005
                                 + 0.607691 w_k1_IBM_s005
                                 + 1.05874 w_k1_INTC_s005
                                 + 0.680466 w_k1_KO_s005
                                 + 0.485428 w_k1_MAT_s005
                                 + 0.922314 w_k1_MSFT_s005
                                 + 0.895432 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.813077 w_k1_ADM_s005
                                 + 0.929568 w_k1_AXP_s005
                                 + 0.990888 w_k1_DIS_s005
                                 + 1.02327 w_k1_FDX_s005
                                 + 1.06155 w_k1_IBM_s005
                                 + 1.13745 w_k1_INTC_s005 + 1.0147 w_k1_KO_s005
                                 + 0.914228 w_k1_MAT_s005
                                 + 1.13354 w_k1_MSFT_s005
                                 + 0.929175 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.940442 w_k1_ADM_s005
                                 + 0.930577 w_k1_AXP_s005
                                 + 1.33056 w_k1_DIS_s005
                                 + 1.08336 w_k1_FDX_s005
                                 + 0.939303 w_k1_IBM_s005
                                 + 0.955487 w_k1_INTC_s005
                                 + 0.954642 w_k1_KO_s005
                                 + 1.09334 w_k1_MAT_s005
                                 + 1.22119 w_k1_MSFT_s005
                                 + 1.03467 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.22741 w_k1_ADM_s005
                                 + 1.12493 w_k1_AXP_s005
                                 + 1.42778 w_k1_DIS_s005 + 1.4065 w_k1_FDX_s005
                                 + 0.745631 w_k1_IBM_s005
                                 + 1.08414 w_k1_INTC_s005
                                 + 0.889867 w_k1_KO_s005
                                 + 0.946952 w_k1_MAT_s005
                                 + 1.16717 w_k1_MSFT_s005
                                 + 1.10891 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.988377 w_k1_ADM_s005
                                 + 1.00151 w_k1_AXP_s005
                                 + 1.31589 w_k1_DIS_s005
                                 + 1.22275 w_k1_FDX_s005 + 0.8599 w_k1_IBM_s005
                                 + 1.31736 w_k1_INTC_s005
                                 + 1.14754 w_k1_KO_s005
                                 + 0.788456 w_k1_MAT_s005
                                 + 1.34494 w_k1_MSFT_s005
                                 + 1.04381 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.964424 w_k1_ADM_s006
                                 + 0.997832 w_k1_AXP_s006
                                 + 1.18594 w_k1_DIS_s006
                                 + 1.13007 w_k1_FDX_s006
                                 + 1.27882 w_k1_IBM_s006
                                 + 1.09058 w_k1_INTC_s006
                                 + 1.13535 w_k1_KO_s006 + 1.18083 w_k1_MAT_s006
                                 + 1.04704 w_k1_MSFT_s006
                                 + 1.13372 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.05886 w_k1_ADM_s006
                                 + 1.51904 w_k1_AXP_s006
                                 + 1.38032 w_k1_DIS_s006
                                 + 1.29262 w_k1_FDX_s006 + 1.1531 w_k1_IBM_s006
                                 + 1.2409 w_k1_INTC_s006 + 1.10913 w_k1_KO_s006
                                 + 1.52229 w_k1_MAT_s006
                                 + 1.10197 w_k1_MSFT_s006
                                 + 1.11721 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.809614 w_k1_ADM_s006
                                 + 0.708692 w_k1_AXP_s006
                                 + 0.762364 w_k1_DIS_s006
                                 + 0.857668 w_k1_FDX_s006
                                 + 0.898874 w_k1_IBM_s006
                                 + 0.965582 w_k1_INTC_s006
                                 + 0.957927 w_k1_KO_s006
                                 + 0.851573 w_k1_MAT_s006
                                 + 0.982725 w_k1_MSFT_s006
                                 + 0.984501 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.818384 w_k1_ADM_s006
                                 + 0.941017 w_k1_AXP_s006
                                 + 0.726932 w_k1_DIS_s006
                                 + 0.814812 w_k1_FDX_s006
                                 + 1.12807 w_k1_IBM_s006
                                 + 0.619933 w_k1_INTC_s006
                                 + 0.925245 w_k1_KO_s006 + 1.6909 w_k1_MAT_s006
                                 + 0.725291 w_k1_MSFT_s006
                                 + 0.989536 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.987473 w_k1_ADM_s006
                                 + 0.989588 w_k1_AXP_s006
                                 + 0.712993 w_k1_DIS_s006 + 1.026 w_k1_FDX_s006
                                 + 0.621717 w_k1_IBM_s006
                                 + 0.98452 w_k1_INTC_s006
                                 + 0.675848 w_k1_KO_s006
                                 + 0.828249 w_k1_MAT_s006
                                 + 0.837139 w_k1_MSFT_s006
                                 + 0.935871 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.732046 w_k1_ADM_s006
                                 + 0.948125 w_k1_AXP_s006
                                 + 0.487385 w_k1_DIS_s006
                                 + 0.795998 w_k1_FDX_s006
                                 + 0.699049 w_k1_IBM_s006
                                 + 0.876171 w_k1_INTC_s006 + 0.626 w_k1_KO_s006
                                 + 0.836932 w_k1_MAT_s006
                                 + 0.707467 w_k1_MSFT_s006
                                 + 0.818042 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.01151 w_k1_ADM_s006
                                 + 0.944701 w_k1_AXP_s006
                                 + 0.978442 w_k1_DIS_s006
                                 + 0.971478 w_k1_FDX_s006
                                 + 1.08222 w_k1_IBM_s006
                                 + 0.956352 w_k1_INTC_s006
                                 + 1.04609 w_k1_KO_s006 + 1.14972 w_k1_MAT_s006
                                 + 1.06639 w_k1_MSFT_s006
                                 + 0.990703 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.16981 w_k1_ADM_s006
                                 + 0.854813 w_k1_AXP_s006
                                 + 0.782809 w_k1_DIS_s006
                                 + 0.935055 w_k1_FDX_s006
                                 + 0.879319 w_k1_IBM_s006
                                 + 0.883963 w_k1_INTC_s006
                                 + 0.849736 w_k1_KO_s006
                                 + 0.758263 w_k1_MAT_s006
                                 + 0.86936 w_k1_MSFT_s006
                                 + 0.951931 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.25616 w_k1_ADM_s007
                                 + 0.858739 w_k1_AXP_s007
                                 + 1.32198 w_k1_DIS_s007
                                 + 1.06894 w_k1_FDX_s007
                                 + 0.981587 w_k1_IBM_s007
                                 + 0.954019 w_k1_INTC_s007
                                 + 1.15407 w_k1_KO_s007 + 1.01673 w_k1_MAT_s007
                                 + 1.21272 w_k1_MSFT_s007 + 1.1675 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.761275 w_k1_ADM_s007
                                 + 0.777831 w_k1_AXP_s007
                                 + 0.71892 w_k1_DIS_s007
                                 + 0.892768 w_k1_FDX_s007
                                 + 0.76345 w_k1_IBM_s007
                                 + 0.844214 w_k1_INTC_s007
                                 + 0.703314 w_k1_KO_s007
                                 + 0.801058 w_k1_MAT_s007
                                 + 0.932223 w_k1_MSFT_s007
                                 + 0.94717 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.29136 w_k1_ADM_s007
                                 + 1.35941 w_k1_AXP_s007
                                 + 1.42664 w_k1_DIS_s007 + 1.2458 w_k1_FDX_s007
                                 + 1.23495 w_k1_IBM_s007
                                 + 1.11446 w_k1_INTC_s007
                                 + 1.12936 w_k1_KO_s007 + 1.57736 w_k1_MAT_s007
                                 + 1.22581 w_k1_MSFT_s007
                                 + 1.21746 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.12743 w_k1_ADM_s007
                                 + 1.18941 w_k1_AXP_s007
                                 + 1.27175 w_k1_DIS_s007
                                 + 1.27813 w_k1_FDX_s007
                                 + 0.70793 w_k1_IBM_s007
                                 + 1.20934 w_k1_INTC_s007
                                 + 0.968008 w_k1_KO_s007
                                 + 0.753941 w_k1_MAT_s007
                                 + 1.21756 w_k1_MSFT_s007
                                 + 1.15065 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.745418 w_k1_ADM_s007
                                 + 0.781464 w_k1_AXP_s007
                                 + 0.69585 w_k1_DIS_s007
                                 + 0.742207 w_k1_FDX_s007
                                 + 1.0036 w_k1_IBM_s007
                                 + 0.786513 w_k1_INTC_s007
                                 + 0.970449 w_k1_KO_s007
                                 + 1.27595 w_k1_MAT_s007
                                 + 0.895787 w_k1_MSFT_s007
                                 + 0.973786 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.934158 w_k1_ADM_s007
                                 + 1.10807 w_k1_AXP_s007
                                 + 1.15389 w_k1_DIS_s007
                                 + 0.984905 w_k1_FDX_s007
                                 + 1.6342 w_k1_IBM_s007
                                 + 0.913294 w_k1_INTC_s007
                                 + 1.36596 w_k1_KO_s007 + 1.84554 w_k1_MAT_s007
                                 + 1.02405 w_k1_MSFT_s007
                                 + 1.18671 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.05299 w_k1_ADM_s007
                                 + 1.04715 w_k1_AXP_s007
                                 + 1.03051 w_k1_DIS_s007
                                 + 1.01617 w_k1_FDX_s007
                                 + 0.930722 w_k1_IBM_s007
                                 + 0.889913 w_k1_INTC_s007
                                 + 1.02186 w_k1_KO_s007 + 1.08913 w_k1_MAT_s007
                                 + 0.931156 w_k1_MSFT_s007
                                 + 1.04091 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.830433 w_k1_ADM_s007
                                 + 0.885366 w_k1_AXP_s007
                                 + 0.908717 w_k1_DIS_s007
                                 + 0.819008 w_k1_FDX_s007
                                 + 1.03859 w_k1_IBM_s007
                                 + 0.843066 w_k1_INTC_s007
                                 + 0.921689 w_k1_KO_s007
                                 + 1.25296 w_k1_MAT_s007
                                 + 0.969928 w_k1_MSFT_s007
                                 + 1.04323 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.920681 w_k1_ADM_s010
                                 + 0.787447 w_k1_AXP_s010
                                 + 0.988995 w_k1_DIS_s010
                                 + 0.907054 w_k1_FDX_s010
                                 + 0.918918 w_k1_IBM_s010
                                 + 0.855194 w_k1_INTC_s010
                                 + 1.05418 w_k1_KO_s010 + 1.36177 w_k1_MAT_s010
                                 + 0.884395 w_k1_MSFT_s010
                                 + 0.928671 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.953013 w_k1_ADM_s010
                                 + 0.899353 w_k1_AXP_s010
                                 + 1.05323 w_k1_DIS_s010
                                 + 0.925026 w_k1_FDX_s010
                                 + 1.31273 w_k1_IBM_s010
                                 + 1.08743 w_k1_INTC_s010
                                 + 1.24796 w_k1_KO_s010 + 1.69125 w_k1_MAT_s010
                                 + 0.950701 w_k1_MSFT_s010
                                 + 0.890581 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.12162 w_k1_ADM_s010
                                 + 1.30007 w_k1_AXP_s010
                                 + 1.14012 w_k1_DIS_s010
                                 + 1.01501 w_k1_FDX_s010
                                 + 1.16409 w_k1_IBM_s010
                                 + 1.17033 w_k1_INTC_s010
                                 + 1.26323 w_k1_KO_s010 + 1.48991 w_k1_MAT_s010
                                 + 1.00795 w_k1_MSFT_s010 + 1.0392 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.1663 w_k1_ADM_s010
                                 + 1.26039 w_k1_AXP_s010
                                 + 0.960321 w_k1_DIS_s010
                                 + 1.1202 w_k1_FDX_s010
                                 + 0.864239 w_k1_IBM_s010
                                 + 1.01963 w_k1_INTC_s010
                                 + 1.10518 w_k1_KO_s010 + 1.25906 w_k1_MAT_s010
                                 + 1.06186 w_k1_MSFT_s010
                                 + 0.979138 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.868522 w_k1_ADM_s010
                                 + 0.708253 w_k1_AXP_s010
                                 + 0.68128 w_k1_DIS_s010
                                 + 0.74936 w_k1_FDX_s010 + 1.0816 w_k1_IBM_s010
                                 + 0.790168 w_k1_INTC_s010
                                 + 0.883301 w_k1_KO_s010
                                 + 1.27789 w_k1_MAT_s010
                                 + 0.733817 w_k1_MSFT_s010
                                 + 0.937303 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.701973 w_k1_ADM_s010
                                 + 0.54731 w_k1_AXP_s010
                                 + 0.629203 w_k1_DIS_s010
                                 + 0.651855 w_k1_FDX_s010
                                 + 0.726412 w_k1_IBM_s010
                                 + 0.648904 w_k1_INTC_s010
                                 + 0.770974 w_k1_KO_s010
                                 + 0.680397 w_k1_MAT_s010
                                 + 0.7143 w_k1_MSFT_s010
                                 + 0.648262 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.652409 w_k1_ADM_s010
                                 + 0.837562 w_k1_AXP_s010
                                 + 0.764618 w_k1_DIS_s010
                                 + 0.770885 w_k1_FDX_s010
                                 + 0.865981 w_k1_IBM_s010
                                 + 0.792788 w_k1_INTC_s010
                                 + 0.893933 w_k1_KO_s010 + 1.1013 w_k1_MAT_s010
                                 + 0.82015 w_k1_MSFT_s010
                                 + 0.809713 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.29685 w_k1_ADM_s010
                                 + 1.11022 w_k1_AXP_s010
                                 + 1.52143 w_k1_DIS_s010 + 1.179 w_k1_FDX_s010
                                 + 1.22415 w_k1_IBM_s010
                                 + 1.18191 w_k1_INTC_s010
                                 + 1.33639 w_k1_KO_s010 + 1.35246 w_k1_MAT_s010
                                 + 1.26821 w_k1_MSFT_s010
                                 + 1.08342 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.903615 w_k1_ADM_s011
                                 + 0.907624 w_k1_AXP_s011
                                 + 1.04392 w_k1_DIS_s011
                                 + 1.01526 w_k1_FDX_s011
                                 + 0.711907 w_k1_IBM_s011
                                 + 1.0831 w_k1_INTC_s011
                                 + 0.872411 w_k1_KO_s011
                                 + 0.667418 w_k1_MAT_s011
                                 + 1.15609 w_k1_MSFT_s011
                                 + 0.935136 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.14269 w_k1_ADM_s011
                                 + 1.81798 w_k1_AXP_s011
                                 + 1.81054 w_k1_DIS_s011
                                 + 1.59664 w_k1_FDX_s011
                                 + 0.967734 w_k1_IBM_s011
                                 + 1.49475 w_k1_INTC_s011
                                 + 1.16609 w_k1_KO_s011 + 1.33867 w_k1_MAT_s011
                                 + 1.61663 w_k1_MSFT_s011
                                 + 1.19102 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.83141 w_k1_ADM_s011
                                 + 1.16753 w_k1_AXP_s011
                                 + 0.947923 w_k1_DIS_s011
                                 + 0.925763 w_k1_FDX_s011
                                 + 1.10014 w_k1_IBM_s011
                                 + 0.914083 w_k1_INTC_s011
                                 + 0.972381 w_k1_KO_s011
                                 + 0.99417 w_k1_MAT_s011
                                 + 0.889838 w_k1_MSFT_s011
                                 + 0.8798 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.956937 w_k1_ADM_s011
                                 + 0.651454 w_k1_AXP_s011
                                 + 0.79998 w_k1_DIS_s011
                                 + 0.92078 w_k1_FDX_s011
                                 + 0.523375 w_k1_IBM_s011
                                 + 0.808609 w_k1_INTC_s011
                                 + 0.620569 w_k1_KO_s011
                                 + 0.40089 w_k1_MAT_s011
                                 + 0.94201 w_k1_MSFT_s011
                                 + 0.884963 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.901068 w_k1_ADM_s011
                                 + 1.26782 w_k1_AXP_s011
                                 + 1.84717 w_k1_DIS_s011
                                 + 1.19543 w_k1_FDX_s011
                                 + 1.70858 w_k1_IBM_s011
                                 + 1.56895 w_k1_INTC_s011
                                 + 1.70699 w_k1_KO_s011 + 1.78856 w_k1_MAT_s011
                                 + 1.49143 w_k1_MSFT_s011
                                 + 1.25804 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.21887 w_k1_ADM_s011
                                 + 1.22775 w_k1_AXP_s011
                                 + 1.26206 w_k1_DIS_s011
                                 + 1.10191 w_k1_FDX_s011
                                 + 1.36567 w_k1_IBM_s011
                                 + 1.24297 w_k1_INTC_s011
                                 + 1.24187 w_k1_KO_s011 + 1.05311 w_k1_MAT_s011
                                 + 1.21767 w_k1_MSFT_s011
                                 + 0.988712 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.872127 w_k1_ADM_s011
                                 + 0.754502 w_k1_AXP_s011
                                 + 0.60844 w_k1_DIS_s011
                                 + 0.754456 w_k1_FDX_s011
                                 + 0.852578 w_k1_IBM_s011
                                 + 0.702719 w_k1_INTC_s011
                                 + 0.715528 w_k1_KO_s011
                                 + 0.883367 w_k1_MAT_s011
                                 + 0.730474 w_k1_MSFT_s011
                                 + 0.81214 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.843744 w_k1_ADM_s011
                                 + 1.10568 w_k1_AXP_s011
                                 + 0.733492 w_k1_DIS_s011
                                 + 0.957366 w_k1_FDX_s011
                                 + 0.668954 w_k1_IBM_s011
                                 + 0.837333 w_k1_INTC_s011
                                 + 0.700585 w_k1_KO_s011
                                 + 0.709096 w_k1_MAT_s011
                                 + 0.891007 w_k1_MSFT_s011
                                 + 0.866485 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.892462 w_k1_ADM_s012
                                 + 0.774224 w_k1_AXP_s012
                                 + 0.723287 w_k1_DIS_s012
                                 + 0.76789 w_k1_FDX_s012
                                 + 0.953227 w_k1_IBM_s012
                                 + 0.918406 w_k1_INTC_s012
                                 + 0.758906 w_k1_KO_s012
                                 + 0.722543 w_k1_MAT_s012
                                 + 0.887607 w_k1_MSFT_s012
                                 + 0.949766 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.782902 w_k1_ADM_s012
                                 + 0.865142 w_k1_AXP_s012
                                 + 1.03674 w_k1_DIS_s012
                                 + 0.851753 w_k1_FDX_s012
                                 + 1.20564 w_k1_IBM_s012
                                 + 0.81151 w_k1_INTC_s012
                                 + 1.04662 w_k1_KO_s012 + 0.91057 w_k1_MAT_s012
                                 + 0.886689 w_k1_MSFT_s012
                                 + 1.07756 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.898454 w_k1_ADM_s012
                                 + 1.13851 w_k1_AXP_s012
                                 + 0.871521 w_k1_DIS_s012
                                 + 0.863101 w_k1_FDX_s012
                                 + 1.27807 w_k1_IBM_s012
                                 + 1.01918 w_k1_INTC_s012
                                 + 1.11713 w_k1_KO_s012 + 1.02505 w_k1_MAT_s012
                                 + 0.882848 w_k1_MSFT_s012
                                 + 0.909292 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.1402 w_k1_ADM_s012
                                 + 1.17713 w_k1_AXP_s012
                                 + 1.19784 w_k1_DIS_s012
                                 + 0.99514 w_k1_FDX_s012
                                 + 1.66089 w_k1_IBM_s012
                                 + 0.914003 w_k1_INTC_s012
                                 + 1.31742 w_k1_KO_s012 + 1.52644 w_k1_MAT_s012
                                 + 0.993 w_k1_MSFT_s012 + 1.17747 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.02114 w_k1_ADM_s012
                                 + 1.06953 w_k1_AXP_s012
                                 + 1.33652 w_k1_DIS_s012
                                 + 1.13381 w_k1_FDX_s012
                                 + 1.01905 w_k1_IBM_s012
                                 + 1.0075 w_k1_INTC_s012 + 1.18939 w_k1_KO_s012
                                 + 1.03308 w_k1_MAT_s012
                                 + 1.12616 w_k1_MSFT_s012
                                 + 1.12147 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.37721 w_k1_ADM_s012
                                 + 1.50061 w_k1_AXP_s012
                                 + 1.44411 w_k1_DIS_s012
                                 + 1.23906 w_k1_FDX_s012 + 1.4255 w_k1_IBM_s012
                                 + 1.26928 w_k1_INTC_s012
                                 + 1.36439 w_k1_KO_s012 + 1.5103 w_k1_MAT_s012
                                 + 1.2177 w_k1_MSFT_s012 + 1.31076 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.03454 w_k1_ADM_s012
                                 + 1.1267 w_k1_AXP_s012 + 1.05691 w_k1_DIS_s012
                                 + 1.03191 w_k1_FDX_s012
                                 + 0.795045 w_k1_IBM_s012
                                 + 0.872244 w_k1_INTC_s012
                                 + 0.878463 w_k1_KO_s012
                                 + 1.06607 w_k1_MAT_s012
                                 + 0.946482 w_k1_MSFT_s012
                                 + 0.980777 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.756324 w_k1_ADM_s012
                                 + 0.638235 w_k1_AXP_s012
                                 + 0.603789 w_k1_DIS_s012
                                 + 0.736131 w_k1_FDX_s012
                                 + 0.611092 w_k1_IBM_s012
                                 + 0.755517 w_k1_INTC_s012
                                 + 0.647799 w_k1_KO_s012
                                 + 0.529448 w_k1_MAT_s012
                                 + 0.731531 w_k1_MSFT_s012
                                 + 0.734829 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.0397 w_k1_ADM_s013
                                 + 0.807718 w_k1_AXP_s013
                                 + 0.754626 w_k1_DIS_s013
                                 + 0.772455 w_k1_FDX_s013
                                 + 0.987143 w_k1_IBM_s013
                                 + 0.989752 w_k1_INTC_s013
                                 + 0.950036 w_k1_KO_s013
                                 + 0.986963 w_k1_MAT_s013
                                 + 0.859947 w_k1_MSFT_s013
                                 + 0.821337 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.20462 w_k1_ADM_s013
                                 + 1.11296 w_k1_AXP_s013
                                 + 1.07967 w_k1_DIS_s013
                                 + 1.04133 w_k1_FDX_s013
                                 + 0.806738 w_k1_IBM_s013
                                 + 1.13447 w_k1_INTC_s013 + 1.0162 w_k1_KO_s013
                                 + 1.01036 w_k1_MAT_s013
                                 + 1.00802 w_k1_MSFT_s013
                                 + 0.915156 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.866593 w_k1_ADM_s013
                                 + 1.0258 w_k1_AXP_s013 + 1.16399 w_k1_DIS_s013
                                 + 0.97687 w_k1_FDX_s013
                                 + 1.33218 w_k1_IBM_s013
                                 + 1.15225 w_k1_INTC_s013
                                 + 1.31086 w_k1_KO_s013 + 1.54024 w_k1_MAT_s013
                                 + 1.06931 w_k1_MSFT_s013
                                 + 0.979656 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.26672 w_k1_ADM_s013
                                 + 0.904621 w_k1_AXP_s013
                                 + 1.01161 w_k1_DIS_s013
                                 + 0.95952 w_k1_FDX_s013
                                 + 0.943062 w_k1_IBM_s013
                                 + 1.00952 w_k1_INTC_s013
                                 + 0.896419 w_k1_KO_s013
                                 + 0.908857 w_k1_MAT_s013
                                 + 1.05501 w_k1_MSFT_s013
                                 + 0.874423 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.70791 w_k1_ADM_s013
                                 + 0.749888 w_k1_AXP_s013
                                 + 0.659697 w_k1_DIS_s013
                                 + 0.693208 w_k1_FDX_s013
                                 + 0.752858 w_k1_IBM_s013
                                 + 0.77866 w_k1_INTC_s013
                                 + 0.680204 w_k1_KO_s013
                                 + 0.693388 w_k1_MAT_s013
                                 + 0.729134 w_k1_MSFT_s013
                                 + 0.714451 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.1486 w_k1_ADM_s013
                                 + 1.42496 w_k1_AXP_s013
                                 + 1.44272 w_k1_DIS_s013
                                 + 1.32744 w_k1_FDX_s013
                                 + 0.767265 w_k1_IBM_s013
                                 + 1.47871 w_k1_INTC_s013
                                 + 1.14539 w_k1_KO_s013 + 1.25777 w_k1_MAT_s013
                                 + 1.39531 w_k1_MSFT_s013
                                 + 1.09224 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.16235 w_k1_ADM_s013
                                 + 1.64131 w_k1_AXP_s013
                                 + 1.30909 w_k1_DIS_s013
                                 + 1.19326 w_k1_FDX_s013
                                 + 1.25122 w_k1_IBM_s013
                                 + 1.3235 w_k1_INTC_s013 + 1.24206 w_k1_KO_s013
                                 + 2.02468 w_k1_MAT_s013
                                 + 1.10559 w_k1_MSFT_s013
                                 + 1.01712 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.32929 w_k1_ADM_s013
                                 + 1.21796 w_k1_AXP_s013
                                 + 1.45527 w_k1_DIS_s013
                                 + 1.18939 w_k1_FDX_s013
                                 + 1.09914 w_k1_IBM_s013
                                 + 1.07672 w_k1_INTC_s013
                                 + 1.23108 w_k1_KO_s013 + 1.36652 w_k1_MAT_s013
                                 + 1.18317 w_k1_MSFT_s013
                                 + 1.08174 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.969607 w_k1_ADM_s014
                                 + 0.758348 w_k1_AXP_s014
                                 + 0.897827 w_k1_DIS_s014
                                 + 0.931984 w_k1_FDX_s014
                                 + 0.816699 w_k1_IBM_s014
                                 + 0.916729 w_k1_INTC_s014
                                 + 0.95579 w_k1_KO_s014 + 0.84186 w_k1_MAT_s014
                                 + 1.0954 w_k1_MSFT_s014
                                 + 0.926366 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.754202 w_k1_ADM_s014
                                 + 0.872654 w_k1_AXP_s014
                                 + 1.1069 w_k1_DIS_s014
                                 + 0.934409 w_k1_FDX_s014
                                 + 1.17188 w_k1_IBM_s014
                                 + 1.16499 w_k1_INTC_s014
                                 + 1.18591 w_k1_KO_s014
                                 + 0.954292 w_k1_MAT_s014
                                 + 1.05991 w_k1_MSFT_s014
                                 + 0.951295 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.805504 w_k1_ADM_s014
                                 + 0.696572 w_k1_AXP_s014
                                 + 0.820424 w_k1_DIS_s014
                                 + 0.727834 w_k1_FDX_s014
                                 + 1.1478 w_k1_IBM_s014
                                 + 0.939029 w_k1_INTC_s014
                                 + 0.914716 w_k1_KO_s014
                                 + 0.908331 w_k1_MAT_s014
                                 + 0.844917 w_k1_MSFT_s014
                                 + 0.836896 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.771007 w_k1_ADM_s014
                                 + 0.705332 w_k1_AXP_s014
                                 + 0.671711 w_k1_DIS_s014
                                 + 0.805583 w_k1_FDX_s014
                                 + 0.712186 w_k1_IBM_s014
                                 + 0.965506 w_k1_INTC_s014
                                 + 0.742716 w_k1_KO_s014
                                 + 0.58611 w_k1_MAT_s014
                                 + 0.747706 w_k1_MSFT_s014
                                 + 0.839122 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.861479 w_k1_ADM_s014
                                 + 1.04459 w_k1_AXP_s014
                                 + 1.11387 w_k1_DIS_s014
                                 + 0.948435 w_k1_FDX_s014
                                 + 1.56437 w_k1_IBM_s014
                                 + 1.1938 w_k1_INTC_s014 + 1.20255 w_k1_KO_s014
                                 + 1.30504 w_k1_MAT_s014
                                 + 1.08892 w_k1_MSFT_s014
                                 + 1.00762 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.11867 w_k1_ADM_s014
                                 + 1.34953 w_k1_AXP_s014
                                 + 1.45304 w_k1_DIS_s014
                                 + 1.07671 w_k1_FDX_s014
                                 + 1.88805 w_k1_IBM_s014
                                 + 1.04829 w_k1_INTC_s014 + 1.5434 w_k1_KO_s014
                                 + 2.15427 w_k1_MAT_s014
                                 + 1.09163 w_k1_MSFT_s014
                                 + 1.33694 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.13006 w_k1_ADM_s014
                                 + 1.13761 w_k1_AXP_s014
                                 + 1.40459 w_k1_DIS_s014
                                 + 1.18494 w_k1_FDX_s014
                                 + 1.06031 w_k1_IBM_s014
                                 + 1.10435 w_k1_INTC_s014 + 1.2629 w_k1_KO_s014
                                 + 1.22344 w_k1_MAT_s014
                                 + 1.23149 w_k1_MSFT_s014
                                 + 1.12027 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.655655 w_k1_ADM_s014
                                 + 1.15363 w_k1_AXP_s014
                                 + 0.88675 w_k1_DIS_s014
                                 + 0.900292 w_k1_FDX_s014
                                 + 1.17645 w_k1_IBM_s014
                                 + 1.24494 w_k1_INTC_s014
                                 + 1.01802 w_k1_KO_s014 + 1.05049 w_k1_MAT_s014
                                 + 1.0053 w_k1_MSFT_s014
                                 + 0.922339 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.892392 w_k1_ADM_s015
                                 + 1.35179 w_k1_AXP_s015
                                 + 1.68568 w_k1_DIS_s015
                                 + 1.38386 w_k1_FDX_s015
                                 + 1.57227 w_k1_IBM_s015
                                 + 1.16835 w_k1_INTC_s015
                                 + 1.46488 w_k1_KO_s015 + 1.43495 w_k1_MAT_s015
                                 + 1.43783 w_k1_MSFT_s015
                                 + 1.19958 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.02268 w_k1_ADM_s015
                                 + 0.784739 w_k1_AXP_s015
                                 + 0.928313 w_k1_DIS_s015
                                 + 0.900525 w_k1_FDX_s015
                                 + 1.07586 w_k1_IBM_s015
                                 + 0.783331 w_k1_INTC_s015
                                 + 0.9485 w_k1_KO_s015 + 0.792455 w_k1_MAT_s015
                                 + 1.02875 w_k1_MSFT_s015
                                 + 0.802609 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.13192 w_k1_ADM_s015
                                 + 0.744463 w_k1_AXP_s015
                                 + 0.809927 w_k1_DIS_s015
                                 + 1.04575 w_k1_FDX_s015
                                 + 0.598682 w_k1_IBM_s015
                                 + 0.686412 w_k1_INTC_s015
                                 + 0.706843 w_k1_KO_s015
                                 + 0.57709 w_k1_MAT_s015
                                 + 0.948899 w_k1_MSFT_s015
                                 + 0.8142 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.843749 w_k1_ADM_s015
                                 + 0.790202 w_k1_AXP_s015
                                 + 0.66181 w_k1_DIS_s015
                                 + 0.86972 w_k1_FDX_s015
                                 + 0.860109 w_k1_IBM_s015
                                 + 0.859892 w_k1_INTC_s015
                                 + 0.735101 w_k1_KO_s015
                                 + 0.955047 w_k1_MAT_s015
                                 + 0.838998 w_k1_MSFT_s015
                                 + 0.791221 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.38236 w_k1_ADM_s015
                                 + 1.2182 w_k1_AXP_s015 + 1.10726 w_k1_DIS_s015
                                 + 1.25262 w_k1_FDX_s015
                                 + 1.08485 w_k1_IBM_s015
                                 + 1.17681 w_k1_INTC_s015
                                 + 1.12702 w_k1_KO_s015
                                 + 0.885943 w_k1_MAT_s015
                                 + 1.22269 w_k1_MSFT_s015
                                 + 1.00155 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.03694 w_k1_ADM_s015
                                 + 1.07762 w_k1_AXP_s015
                                 + 1.25211 w_k1_DIS_s015
                                 + 1.16809 w_k1_FDX_s015
                                 + 0.88326 w_k1_IBM_s015
                                 + 1.21166 w_k1_INTC_s015
                                 + 1.01387 w_k1_KO_s015
                                 + 0.703596 w_k1_MAT_s015
                                 + 1.23017 w_k1_MSFT_s015
                                 + 1.01563 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.616499 w_k1_ADM_s015
                                 + 0.80532 w_k1_AXP_s015
                                 + 0.875374 w_k1_DIS_s015
                                 + 0.844158 w_k1_FDX_s015
                                 + 1.35543 w_k1_IBM_s015
                                 + 0.858811 w_k1_INTC_s015
                                 + 1.04248 w_k1_KO_s015 + 1.36952 w_k1_MAT_s015
                                 + 0.916978 w_k1_MSFT_s015
                                 + 0.917265 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.722622 w_k1_ADM_s015
                                 + 0.586722 w_k1_AXP_s015
                                 + 0.662023 w_k1_DIS_s015
                                 + 0.798397 w_k1_FDX_s015
                                 + 0.72879 w_k1_IBM_s015
                                 + 0.808783 w_k1_INTC_s015
                                 + 0.667097 w_k1_KO_s015
                                 + 0.634902 w_k1_MAT_s015
                                 + 0.832808 w_k1_MSFT_s015
                                 + 0.660753 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.967214 w_k1_ADM_s016
                                 + 0.836868 w_k1_AXP_s016
                                 + 0.80633 w_k1_DIS_s016
                                 + 0.956506 w_k1_FDX_s016
                                 + 0.67124 w_k1_IBM_s016
                                 + 0.846761 w_k1_INTC_s016
                                 + 0.851071 w_k1_KO_s016
                                 + 0.861711 w_k1_MAT_s016
                                 + 1.03013 w_k1_MSFT_s016
                                 + 0.908355 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.753358 w_k1_ADM_s016
                                 + 0.63384 w_k1_AXP_s016
                                 + 0.455697 w_k1_DIS_s016
                                 + 0.645685 w_k1_FDX_s016
                                 + 0.668299 w_k1_IBM_s016
                                 + 0.70855 w_k1_INTC_s016
                                 + 0.733278 w_k1_KO_s016
                                 + 0.601424 w_k1_MAT_s016
                                 + 0.679244 w_k1_MSFT_s016
                                 + 0.796876 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.42526 w_k1_ADM_s016
                                 + 1.3707 w_k1_AXP_s016 + 1.45505 w_k1_DIS_s016
                                 + 1.19981 w_k1_FDX_s016
                                 + 1.37536 w_k1_IBM_s016
                                 + 1.36893 w_k1_INTC_s016 + 1.4307 w_k1_KO_s016
                                 + 1.84753 w_k1_MAT_s016
                                 + 1.23037 w_k1_MSFT_s016
                                 + 1.29458 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.05845 w_k1_ADM_s016
                                 + 0.729249 w_k1_AXP_s016
                                 + 0.64774 w_k1_DIS_s016
                                 + 0.714748 w_k1_FDX_s016
                                 + 0.924488 w_k1_IBM_s016
                                 + 0.731681 w_k1_INTC_s016
                                 + 0.995343 w_k1_KO_s016
                                 + 0.815188 w_k1_MAT_s016
                                 + 0.799556 w_k1_MSFT_s016
                                 + 0.979534 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.48729 w_k1_ADM_s016
                                 + 1.21568 w_k1_AXP_s016
                                 + 1.03764 w_k1_DIS_s016
                                 + 1.18823 w_k1_FDX_s016
                                 + 0.856568 w_k1_IBM_s016
                                 + 0.960235 w_k1_INTC_s016
                                 + 1.01468 w_k1_KO_s016 + 1.0499 w_k1_MAT_s016
                                 + 1.0875 w_k1_MSFT_s016 + 1.1636 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.38176 w_k1_ADM_s016
                                 + 1.09723 w_k1_AXP_s016
                                 + 1.30316 w_k1_DIS_s016
                                 + 1.12708 w_k1_FDX_s016
                                 + 0.851874 w_k1_IBM_s016
                                 + 0.935342 w_k1_INTC_s016
                                 + 1.22902 w_k1_KO_s016 + 1.17731 w_k1_MAT_s016
                                 + 1.03818 w_k1_MSFT_s016
                                 + 1.08255 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.06052 w_k1_ADM_s016
                                 + 0.921584 w_k1_AXP_s016
                                 + 0.946199 w_k1_DIS_s016
                                 + 0.940259 w_k1_FDX_s016
                                 + 1.04227 w_k1_IBM_s016
                                 + 1.05088 w_k1_INTC_s016
                                 + 1.03744 w_k1_KO_s016 + 1.236 w_k1_MAT_s016
                                 + 0.928899 w_k1_MSFT_s016
                                 + 1.07307 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.69288 w_k1_ADM_s016
                                 + 1.33881 w_k1_AXP_s016
                                 + 2.00979 w_k1_DIS_s016
                                 + 1.46029 w_k1_FDX_s016
                                 + 1.11817 w_k1_IBM_s016
                                 + 1.50784 w_k1_INTC_s016
                                 + 1.61187 w_k1_KO_s016 + 1.15388 w_k1_MAT_s016
                                 + 1.68035 w_k1_MSFT_s016
                                 + 1.35409 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.35005 w_k1_ADM_s017
                                 + 1.4667 w_k1_AXP_s017 + 1.71723 w_k1_DIS_s017
                                 + 1.38638 w_k1_FDX_s017
                                 + 1.18588 w_k1_IBM_s017
                                 + 1.31911 w_k1_INTC_s017
                                 + 1.19178 w_k1_KO_s017 + 1.10062 w_k1_MAT_s017
                                 + 1.23663 w_k1_MSFT_s017
                                 + 1.27395 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.890843 w_k1_ADM_s017
                                 + 0.769758 w_k1_AXP_s017
                                 + 0.613691 w_k1_DIS_s017
                                 + 0.768689 w_k1_FDX_s017
                                 + 0.638863 w_k1_IBM_s017
                                 + 1.0274 w_k1_INTC_s017 + 0.64864 w_k1_KO_s017
                                 + 0.527746 w_k1_MAT_s017
                                 + 0.833455 w_k1_MSFT_s017
                                 + 0.79638 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.89655 w_k1_ADM_s017
                                 + 0.926216 w_k1_AXP_s017
                                 + 1.11041 w_k1_DIS_s017 + 1.0158 w_k1_FDX_s017
                                 + 1.03808 w_k1_IBM_s017
                                 + 1.10112 w_k1_INTC_s017
                                 + 1.15935 w_k1_KO_s017
                                 + 0.867675 w_k1_MAT_s017
                                 + 1.11118 w_k1_MSFT_s017
                                 + 0.998416 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.699211 w_k1_ADM_s017
                                 + 0.924711 w_k1_AXP_s017
                                 + 0.787952 w_k1_DIS_s017
                                 + 0.806256 w_k1_FDX_s017
                                 + 0.9126 w_k1_IBM_s017
                                 + 1.15685 w_k1_INTC_s017
                                 + 0.837331 w_k1_KO_s017
                                 + 0.793898 w_k1_MAT_s017
                                 + 0.910646 w_k1_MSFT_s017
                                 + 0.926755 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.971459 w_k1_ADM_s017
                                 + 1.14981 w_k1_AXP_s017
                                 + 1.52414 w_k1_DIS_s017
                                 + 1.02196 w_k1_FDX_s017
                                 + 1.72275 w_k1_IBM_s017
                                 + 1.3363 w_k1_INTC_s017 + 1.36824 w_k1_KO_s017
                                 + 1.41691 w_k1_MAT_s017
                                 + 1.16774 w_k1_MSFT_s017
                                 + 1.19063 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.941891 w_k1_ADM_s017
                                 + 1.30823 w_k1_AXP_s017
                                 + 0.89816 w_k1_DIS_s017
                                 + 0.947417 w_k1_FDX_s017
                                 + 1.34444 w_k1_IBM_s017
                                 + 1.15941 w_k1_INTC_s017
                                 + 0.931003 w_k1_KO_s017
                                 + 1.25498 w_k1_MAT_s017
                                 + 0.901831 w_k1_MSFT_s017
                                 + 1.04943 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.22945 w_k1_ADM_s017
                                 + 1.36084 w_k1_AXP_s017
                                 + 1.26287 w_k1_DIS_s017
                                 + 1.15622 w_k1_FDX_s017
                                 + 1.29846 w_k1_IBM_s017
                                 + 1.31352 w_k1_INTC_s017
                                 + 1.25865 w_k1_KO_s017 + 1.35758 w_k1_MAT_s017
                                 + 1.20951 w_k1_MSFT_s017
                                 + 1.27027 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.04059 w_k1_ADM_s017
                                 + 0.980679 w_k1_AXP_s017
                                 + 1.38912 w_k1_DIS_s017 + 1.0978 w_k1_FDX_s017
                                 + 0.921908 w_k1_IBM_s017
                                 + 1.09113 w_k1_INTC_s017
                                 + 1.00902 w_k1_KO_s017 + 1.00816 w_k1_MAT_s017
                                 + 1.26494 w_k1_MSFT_s017
                                 + 1.05756 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.939374 w_k1_ADM_s100
                                 + 1.13156 w_k1_AXP_s100
                                 + 1.01737 w_k1_DIS_s100
                                 + 1.07037 w_k1_FDX_s100
                                 + 1.05463 w_k1_IBM_s100
                                 + 1.11131 w_k1_INTC_s100
                                 + 1.03307 w_k1_KO_s100 + 1.21307 w_k1_MAT_s100
                                 + 0.999978 w_k1_MSFT_s100
                                 + 1.02589 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.21236 w_k1_ADM_s100
                                 + 1.28385 w_k1_AXP_s100
                                 + 1.41879 w_k1_DIS_s100 + 1.2468 w_k1_FDX_s100
                                 + 1.38048 w_k1_IBM_s100
                                 + 1.32309 w_k1_INTC_s100
                                 + 1.20085 w_k1_KO_s100 + 1.45344 w_k1_MAT_s100
                                 + 1.18517 w_k1_MSFT_s100
                                 + 1.23039 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.43579 w_k1_ADM_s100
                                 + 1.42346 w_k1_AXP_s100
                                 + 1.78189 w_k1_DIS_s100
                                 + 1.47864 w_k1_FDX_s100
                                 + 1.08501 w_k1_IBM_s100
                                 + 1.3849 w_k1_INTC_s100 + 1.24694 w_k1_KO_s100
                                 + 1.11677 w_k1_MAT_s100
                                 + 1.25128 w_k1_MSFT_s100 + 1.194 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.848983 w_k1_ADM_s100
                                 + 0.668853 w_k1_AXP_s100
                                 + 0.737666 w_k1_DIS_s100
                                 + 0.872216 w_k1_FDX_s100
                                 + 0.669888 w_k1_IBM_s100
                                 + 0.894709 w_k1_INTC_s100
                                 + 0.737663 w_k1_KO_s100
                                 + 0.489136 w_k1_MAT_s100
                                 + 0.810238 w_k1_MSFT_s100
                                 + 0.911182 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09711 w_k1_ADM_s100
                                 + 0.911967 w_k1_AXP_s100
                                 + 1.20684 w_k1_DIS_s100
                                 + 1.24818 w_k1_FDX_s100
                                 + 0.744499 w_k1_IBM_s100
                                 + 0.937606 w_k1_INTC_s100
                                 + 0.870964 w_k1_KO_s100
                                 + 0.890259 w_k1_MAT_s100
                                 + 1.1306 w_k1_MSFT_s100 + 1.06387 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.884672 w_k1_ADM_s100
                                 + 0.940462 w_k1_AXP_s100
                                 + 0.963818 w_k1_DIS_s100
                                 + 0.879112 w_k1_FDX_s100
                                 + 1.37527 w_k1_IBM_s100
                                 + 0.983104 w_k1_INTC_s100
                                 + 1.0033 w_k1_KO_s100 + 1.26351 w_k1_MAT_s100
                                 + 0.792199 w_k1_MSFT_s100
                                 + 1.04924 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.618366 w_k1_ADM_s100
                                 + 0.557943 w_k1_AXP_s100
                                 + 0.471393 w_k1_DIS_s100
                                 + 0.610197 w_k1_FDX_s100
                                 + 0.775389 w_k1_IBM_s100
                                 + 0.852058 w_k1_INTC_s100
                                 + 0.67739 w_k1_KO_s100
                                 + 0.695868 w_k1_MAT_s100
                                 + 0.593541 w_k1_MSFT_s100
                                 + 0.778964 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.805771 w_k1_ADM_s100
                                 + 0.786808 w_k1_AXP_s100
                                 + 0.857386 w_k1_DIS_s100
                                 + 0.912762 w_k1_FDX_s100
                                 + 0.969328 w_k1_IBM_s100
                                 + 0.990508 w_k1_INTC_s100
                                 + 0.90715 w_k1_KO_s100 + 1.05148 w_k1_MAT_s100
                                 + 0.85471 w_k1_MSFT_s100
                                 + 0.922274 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.724185 w_k1_ADM_s101
                                 + 0.701623 w_k1_AXP_s101
                                 + 0.596101 w_k1_DIS_s101
                                 + 0.766753 w_k1_FDX_s101
                                 + 0.613176 w_k1_IBM_s101
                                 + 0.768443 w_k1_INTC_s101
                                 + 0.752083 w_k1_KO_s101
                                 + 0.747629 w_k1_MAT_s101
                                 + 0.612006 w_k1_MSFT_s101
                                 + 0.826575 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.05794 w_k1_ADM_s101
                                 + 1.03445 w_k1_AXP_s101
                                 + 0.989025 w_k1_DIS_s101
                                 + 1.0738 w_k1_FDX_s101
                                 + 0.752701 w_k1_IBM_s101
                                 + 0.982186 w_k1_INTC_s101
                                 + 1.0648 w_k1_KO_s101 + 0.92753 w_k1_MAT_s101
                                 + 0.851814 w_k1_MSFT_s101
                                 + 0.977695 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.6806 w_k1_ADM_s101
                                 + 0.746853 w_k1_AXP_s101
                                 + 0.624193 w_k1_DIS_s101
                                 + 0.750182 w_k1_FDX_s101
                                 + 0.866946 w_k1_IBM_s101
                                 + 0.780424 w_k1_INTC_s101
                                 + 0.852638 w_k1_KO_s101
                                 + 0.940533 w_k1_MAT_s101
                                 + 0.778772 w_k1_MSFT_s101
                                 + 0.885258 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09183 w_k1_ADM_s101
                                 + 1.07964 w_k1_AXP_s101
                                 + 1.30285 w_k1_DIS_s101
                                 + 1.12238 w_k1_FDX_s101
                                 + 1.16966 w_k1_IBM_s101
                                 + 1.20611 w_k1_INTC_s101
                                 + 1.15944 w_k1_KO_s101 + 1.11778 w_k1_MAT_s101
                                 + 1.11999 w_k1_MSFT_s101
                                 + 0.988605 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.56597 w_k1_ADM_s101
                                 + 1.36918 w_k1_AXP_s101
                                 + 1.56052 w_k1_DIS_s101
                                 + 1.25628 w_k1_FDX_s101
                                 + 1.71293 w_k1_IBM_s101
                                 + 1.18467 w_k1_INTC_s101
                                 + 1.61988 w_k1_KO_s101 + 1.84246 w_k1_MAT_s101
                                 + 1.2863 w_k1_MSFT_s101 + 1.23912 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.820966 w_k1_ADM_s101
                                 + 1.00948 w_k1_AXP_s101
                                 + 0.858227 w_k1_DIS_s101 + 0.899 w_k1_FDX_s101
                                 + 1.06359 w_k1_IBM_s101
                                 + 0.843389 w_k1_INTC_s101
                                 + 0.954078 w_k1_KO_s101
                                 + 1.43537 w_k1_MAT_s101
                                 + 0.813617 w_k1_MSFT_s101
                                 + 0.985001 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.766111 w_k1_ADM_s101
                                 + 1.02855 w_k1_AXP_s101
                                 + 0.941491 w_k1_DIS_s101
                                 + 0.823932 w_k1_FDX_s101
                                 + 1.57518 w_k1_IBM_s101
                                 + 0.797349 w_k1_INTC_s101
                                 + 1.21592 w_k1_KO_s101 + 1.84939 w_k1_MAT_s101
                                 + 0.806164 w_k1_MSFT_s101
                                 + 1.01165 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.31687 w_k1_ADM_s101
                                 + 1.26643 w_k1_AXP_s101
                                 + 1.69299 w_k1_DIS_s101
                                 + 1.48774 w_k1_FDX_s101
                                 + 0.867443 w_k1_IBM_s101
                                 + 1.22522 w_k1_INTC_s101
                                 + 1.21576 w_k1_KO_s101
                                 + 0.955788 w_k1_MAT_s101
                                 + 1.31474 w_k1_MSFT_s101
                                 + 1.05947 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07063 w_k1_ADM_s102
                                 + 1.10989 w_k1_AXP_s102
                                 + 0.922714 w_k1_DIS_s102
                                 + 0.965788 w_k1_FDX_s102
                                 + 1.19792 w_k1_IBM_s102
                                 + 1.09264 w_k1_INTC_s102
                                 + 1.06288 w_k1_KO_s102 + 1.34084 w_k1_MAT_s102
                                 + 0.926276 w_k1_MSFT_s102
                                 + 0.980546 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.30719 w_k1_ADM_s102
                                 + 1.43281 w_k1_AXP_s102
                                 + 1.32727 w_k1_DIS_s102
                                 + 1.25087 w_k1_FDX_s102
                                 + 1.49002 w_k1_IBM_s102
                                 + 1.29307 w_k1_INTC_s102
                                 + 1.18623 w_k1_KO_s102 + 1.4097 w_k1_MAT_s102
                                 + 1.11554 w_k1_MSFT_s102
                                 + 1.04217 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.735824 w_k1_ADM_s102
                                 + 0.764908 w_k1_AXP_s102
                                 + 0.56522 w_k1_DIS_s102
                                 + 0.64202 w_k1_FDX_s102
                                 + 0.943874 w_k1_IBM_s102
                                 + 0.823835 w_k1_INTC_s102
                                 + 0.76265 w_k1_KO_s102
                                 + 0.889172 w_k1_MAT_s102
                                 + 0.640482 w_k1_MSFT_s102
                                 + 0.802979 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.990801 w_k1_ADM_s102
                                 + 0.869514 w_k1_AXP_s102
                                 + 1.05684 w_k1_DIS_s102
                                 + 0.893126 w_k1_FDX_s102
                                 + 1.24848 w_k1_IBM_s102
                                 + 1.16392 w_k1_INTC_s102
                                 + 1.13299 w_k1_KO_s102
                                 + 0.946161 w_k1_MAT_s102
                                 + 0.999367 w_k1_MSFT_s102
                                 + 1.01165 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.877926 w_k1_ADM_s102
                                 + 0.851563 w_k1_AXP_s102
                                 + 0.861514 w_k1_DIS_s102
                                 + 0.935575 w_k1_FDX_s102
                                 + 0.828422 w_k1_IBM_s102
                                 + 1.01152 w_k1_INTC_s102
                                 + 0.886832 w_k1_KO_s102
                                 + 0.867384 w_k1_MAT_s102
                                 + 1.00144 w_k1_MSFT_s102
                                 + 0.891436 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.24426 w_k1_ADM_s102
                                 + 1.28655 w_k1_AXP_s102
                                 + 1.52599 w_k1_DIS_s102
                                 + 1.28002 w_k1_FDX_s102
                                 + 1.18492 w_k1_IBM_s102
                                 + 1.30821 w_k1_INTC_s102
                                 + 1.30813 w_k1_KO_s102
                                 + 0.900242 w_k1_MAT_s102
                                 + 1.23427 w_k1_MSFT_s102
                                 + 1.22645 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.830834 w_k1_ADM_s102
                                 + 0.695751 w_k1_AXP_s102
                                 + 0.609621 w_k1_DIS_s102
                                 + 0.765511 w_k1_FDX_s102
                                 + 0.660043 w_k1_IBM_s102
                                 + 0.720674 w_k1_INTC_s102
                                 + 0.715544 w_k1_KO_s102
                                 + 0.604375 w_k1_MAT_s102
                                 + 0.694324 w_k1_MSFT_s102
                                 + 0.773278 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.04883 w_k1_ADM_s102
                                 + 1.24754 w_k1_AXP_s102
                                 + 1.31059 w_k1_DIS_s102
                                 + 1.32371 w_k1_FDX_s102
                                 + 0.908112 w_k1_IBM_s102
                                 + 1.38032 w_k1_INTC_s102
                                 + 0.97627 w_k1_KO_s102
                                 + 0.816998 w_k1_MAT_s102
                                 + 1.13008 w_k1_MSFT_s102
                                 + 1.11021 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.27185 w_k1_ADM_s103
                                 + 1.2382 w_k1_AXP_s103 + 1.50614 w_k1_DIS_s103
                                 + 1.32563 w_k1_FDX_s103
                                 + 1.02668 w_k1_IBM_s103
                                 + 1.18687 w_k1_INTC_s103
                                 + 1.12077 w_k1_KO_s103 + 1.19887 w_k1_MAT_s103
                                 + 1.42151 w_k1_MSFT_s103
                                 + 1.16331 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.09718 w_k1_ADM_s103
                                 + 1.02041 w_k1_AXP_s103
                                 + 0.844971 w_k1_DIS_s103
                                 + 0.996446 w_k1_FDX_s103
                                 + 0.973402 w_k1_IBM_s103
                                 + 0.94423 w_k1_INTC_s103
                                 + 0.76008 w_k1_KO_s103
                                 + 0.820473 w_k1_MAT_s103
                                 + 0.981685 w_k1_MSFT_s103
                                 + 0.939648 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.47086 w_k1_ADM_s103
                                 + 1.34955 w_k1_AXP_s103 + 1.2665 w_k1_DIS_s103
                                 + 1.27997 w_k1_FDX_s103
                                 + 1.17516 w_k1_IBM_s103
                                 + 1.19877 w_k1_INTC_s103
                                 + 1.06451 w_k1_KO_s103 + 0.80994 w_k1_MAT_s103
                                 + 1.17685 w_k1_MSFT_s103
                                 + 1.22943 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.839958 w_k1_ADM_s103
                                 + 0.761164 w_k1_AXP_s103
                                 + 0.704213 w_k1_DIS_s103
                                 + 0.822363 w_k1_FDX_s103
                                 + 0.782696 w_k1_IBM_s103
                                 + 0.846031 w_k1_INTC_s103
                                 + 0.770314 w_k1_KO_s103
                                 + 0.779101 w_k1_MAT_s103
                                 + 0.855011 w_k1_MSFT_s103
                                 + 0.96227 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.27949 w_k1_ADM_s103
                                 + 1.19356 w_k1_AXP_s103
                                 + 1.09271 w_k1_DIS_s103
                                 + 1.15309 w_k1_FDX_s103
                                 + 0.867977 w_k1_IBM_s103
                                 + 0.988095 w_k1_INTC_s103
                                 + 0.996963 w_k1_KO_s103
                                 + 0.938057 w_k1_MAT_s103
                                 + 1.08837 w_k1_MSFT_s103
                                 + 1.09878 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.16637 w_k1_ADM_s103
                                 + 0.956232 w_k1_AXP_s103
                                 + 1.00754 w_k1_DIS_s103
                                 + 0.98909 w_k1_FDX_s103
                                 + 0.818046 w_k1_IBM_s103
                                 + 0.968626 w_k1_INTC_s103
                                 + 0.943366 w_k1_KO_s103
                                 + 0.771373 w_k1_MAT_s103
                                 + 1.01885 w_k1_MSFT_s103
                                 + 0.995287 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.992196 w_k1_ADM_s103
                                 + 0.777046 w_k1_AXP_s103
                                 + 0.867096 w_k1_DIS_s103
                                 + 0.872406 w_k1_FDX_s103
                                 + 1.21037 w_k1_IBM_s103
                                 + 0.948611 w_k1_INTC_s103
                                 + 1.02304 w_k1_KO_s103 + 1.22187 w_k1_MAT_s103
                                 + 0.999439 w_k1_MSFT_s103
                                 + 1.00223 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.1806 w_k1_ADM_s103
                                 + 1.18914 w_k1_AXP_s103 + 1.4378 w_k1_DIS_s103
                                 + 1.18231 w_k1_FDX_s103 + 1.3549 w_k1_IBM_s103
                                 + 1.19903 w_k1_INTC_s103
                                 + 1.26664 w_k1_KO_s103 + 1.19437 w_k1_MAT_s103
                                 + 1.1977 w_k1_MSFT_s103 + 1.24148 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.854657 w_k1_ADM_s104
                                 + 0.641316 w_k1_AXP_s104
                                 + 0.717825 w_k1_DIS_s104
                                 + 0.703978 w_k1_FDX_s104
                                 + 0.937774 w_k1_IBM_s104
                                 + 0.968799 w_k1_INTC_s104
                                 + 0.781501 w_k1_KO_s104
                                 + 0.879812 w_k1_MAT_s104
                                 + 0.716494 w_k1_MSFT_s104
                                 + 0.848636 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.824859 w_k1_ADM_s104
                                 + 0.806269 w_k1_AXP_s104
                                 + 0.729366 w_k1_DIS_s104
                                 + 0.680791 w_k1_FDX_s104
                                 + 1.43716 w_k1_IBM_s104
                                 + 0.980808 w_k1_INTC_s104
                                 + 0.939433 w_k1_KO_s104
                                 + 1.12093 w_k1_MAT_s104
                                 + 0.754613 w_k1_MSFT_s104
                                 + 0.968816 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.10966 w_k1_ADM_s104
                                 + 0.955295 w_k1_AXP_s104
                                 + 1.24453 w_k1_DIS_s104
                                 + 1.17451 w_k1_FDX_s104
                                 + 0.653636 w_k1_IBM_s104
                                 + 1.1339 w_k1_INTC_s104
                                 + 0.897253 w_k1_KO_s104
                                 + 0.552533 w_k1_MAT_s104
                                 + 1.10695 w_k1_MSFT_s104
                                 + 1.02825 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.10288 w_k1_ADM_s104
                                 + 1.00902 w_k1_AXP_s104
                                 + 0.972823 w_k1_DIS_s104
                                 + 1.02887 w_k1_FDX_s104
                                 + 0.928864 w_k1_IBM_s104
                                 + 1.00763 w_k1_INTC_s104
                                 + 0.869881 w_k1_KO_s104
                                 + 0.965798 w_k1_MAT_s104
                                 + 0.846554 w_k1_MSFT_s104
                                 + 1.00637 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.840459 w_k1_ADM_s104
                                 + 0.798766 w_k1_AXP_s104
                                 + 0.68401 w_k1_DIS_s104
                                 + 0.933587 w_k1_FDX_s104
                                 + 0.633168 w_k1_IBM_s104
                                 + 1.06226 w_k1_INTC_s104
                                 + 0.673918 w_k1_KO_s104
                                 + 0.522094 w_k1_MAT_s104
                                 + 0.78522 w_k1_MSFT_s104
                                 + 0.817048 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.29234 w_k1_ADM_s104
                                 + 1.51019 w_k1_AXP_s104
                                 + 1.87181 w_k1_DIS_s104
                                 + 1.40037 w_k1_FDX_s104
                                 + 1.26537 w_k1_IBM_s104
                                 + 1.59312 w_k1_INTC_s104
                                 + 1.35206 w_k1_KO_s104
                                 + 0.828852 w_k1_MAT_s104
                                 + 1.32453 w_k1_MSFT_s104
                                 + 1.19428 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.831137 w_k1_ADM_s104
                                 + 0.911627 w_k1_AXP_s104
                                 + 0.934975 w_k1_DIS_s104
                                 + 0.923647 w_k1_FDX_s104
                                 + 1.35359 w_k1_IBM_s104
                                 + 1.17061 w_k1_INTC_s104
                                 + 1.18343 w_k1_KO_s104 + 1.16739 w_k1_MAT_s104
                                 + 0.89418 w_k1_MSFT_s104
                                 + 0.987895 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.00613 w_k1_ADM_s104
                                 + 1.26513 w_k1_AXP_s104
                                 + 1.29787 w_k1_DIS_s104 + 1.176 w_k1_FDX_s104
                                 + 1.17183 w_k1_IBM_s104
                                 + 1.4295 w_k1_INTC_s104 + 1.13318 w_k1_KO_s104
                                 + 1.06722 w_k1_MAT_s104
                                 + 1.17627 w_k1_MSFT_s104
                                 + 1.03913 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.693192 w_k1_ADM_s105
                                 + 0.761788 w_k1_AXP_s105
                                 + 0.8351 w_k1_DIS_s105
                                 + 0.745181 w_k1_FDX_s105
                                 + 1.17816 w_k1_IBM_s105
                                 + 1.21335 w_k1_INTC_s105
                                 + 1.01213 w_k1_KO_s105
                                 + 0.957741 w_k1_MAT_s105
                                 + 0.772171 w_k1_MSFT_s105
                                 + 0.909678 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.891431 w_k1_ADM_s105
                                 + 0.927006 w_k1_AXP_s105
                                 + 0.795123 w_k1_DIS_s105
                                 + 0.865837 w_k1_FDX_s105
                                 + 0.998754 w_k1_IBM_s105
                                 + 1.03582 w_k1_INTC_s105
                                 + 0.899192 w_k1_KO_s105
                                 + 1.11545 w_k1_MAT_s105
                                 + 0.862034 w_k1_MSFT_s105
                                 + 0.939115 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.07897 w_k1_ADM_s105
                                 + 1.00004 w_k1_AXP_s105
                                 + 1.02602 w_k1_DIS_s105 + 1.1631 w_k1_FDX_s105
                                 + 0.805784 w_k1_IBM_s105
                                 + 1.38795 w_k1_INTC_s105
                                 + 0.887277 w_k1_KO_s105
                                 + 1.04262 w_k1_MAT_s105
                                 + 0.993729 w_k1_MSFT_s105
                                 + 0.946031 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.19176 w_k1_ADM_s105
                                 + 1.14111 w_k1_AXP_s105
                                 + 1.46705 w_k1_DIS_s105
                                 + 1.28236 w_k1_FDX_s105
                                 + 1.18458 w_k1_IBM_s105
                                 + 1.44589 w_k1_INTC_s105
                                 + 1.20478 w_k1_KO_s105 + 1.13861 w_k1_MAT_s105
                                 + 1.24415 w_k1_MSFT_s105
                                 + 1.15898 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.876765 w_k1_ADM_s105
                                 + 0.733165 w_k1_AXP_s105
                                 + 0.701294 w_k1_DIS_s105
                                 + 0.871751 w_k1_FDX_s105
                                 + 0.704769 w_k1_IBM_s105
                                 + 1.13737 w_k1_INTC_s105
                                 + 0.800173 w_k1_KO_s105
                                 + 0.674171 w_k1_MAT_s105
                                 + 0.967153 w_k1_MSFT_s105
                                 + 0.79443 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.854896 w_k1_ADM_s105
                                 + 0.982309 w_k1_AXP_s105
                                 + 1.12175 w_k1_DIS_s105
                                 + 0.918096 w_k1_FDX_s105
                                 + 1.73416 w_k1_IBM_s105
                                 + 1.18366 w_k1_INTC_s105
                                 + 1.22025 w_k1_KO_s105 + 1.927 w_k1_MAT_s105
                                 + 1.07896 w_k1_MSFT_s105
                                 + 0.987864 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.908679 w_k1_ADM_s105
                                 + 1.20637 w_k1_AXP_s105
                                 + 1.38763 w_k1_DIS_s105
                                 + 1.18024 w_k1_FDX_s105
                                 + 1.47495 w_k1_IBM_s105
                                 + 1.38379 w_k1_INTC_s105
                                 + 1.31967 w_k1_KO_s105 + 1.69855 w_k1_MAT_s105
                                 + 1.17122 w_k1_MSFT_s105
                                 + 1.04615 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.15076 w_k1_ADM_s105
                                 + 0.990906 w_k1_AXP_s105
                                 + 1.54297 w_k1_DIS_s105
                                 + 1.31722 w_k1_FDX_s105
                                 + 0.895305 w_k1_IBM_s105
                                 + 1.34144 w_k1_INTC_s105 + 1.051 w_k1_KO_s105
                                 + 1.13821 w_k1_MAT_s105
                                 + 1.23068 w_k1_MSFT_s105 + 1.0392 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.834299 w_k1_ADM_s106
                                 + 0.657233 w_k1_AXP_s106
                                 + 0.76462 w_k1_DIS_s106
                                 + 0.81888 w_k1_FDX_s106 + 1.0879 w_k1_IBM_s106
                                 + 0.827035 w_k1_INTC_s106
                                 + 0.978596 w_k1_KO_s106
                                 + 0.911036 w_k1_MAT_s106
                                 + 0.870886 w_k1_MSFT_s106
                                 + 0.937515 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.72067 w_k1_ADM_s106
                                 + 0.566457 w_k1_AXP_s106
                                 + 0.56484 w_k1_DIS_s106
                                 + 0.706395 w_k1_FDX_s106
                                 + 0.637876 w_k1_IBM_s106
                                 + 0.664654 w_k1_INTC_s106
                                 + 0.566629 w_k1_KO_s106
                                 + 0.478564 w_k1_MAT_s106
                                 + 0.789217 w_k1_MSFT_s106
                                 + 0.745115 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12904 w_k1_ADM_s106
                                 + 0.795019 w_k1_AXP_s106
                                 + 1.14467 w_k1_DIS_s106
                                 + 0.989465 w_k1_FDX_s106
                                 + 1.03515 w_k1_IBM_s106
                                 + 1.13457 w_k1_INTC_s106
                                 + 1.15395 w_k1_KO_s106
                                 + 0.806152 w_k1_MAT_s106
                                 + 1.18488 w_k1_MSFT_s106
                                 + 1.09662 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.919547 w_k1_ADM_s106
                                 + 0.721307 w_k1_AXP_s106
                                 + 0.865017 w_k1_DIS_s106
                                 + 0.97891 w_k1_FDX_s106
                                 + 0.67472 w_k1_IBM_s106
                                 + 0.90946 w_k1_INTC_s106
                                 + 0.888111 w_k1_KO_s106
                                 + 0.697694 w_k1_MAT_s106
                                 + 1.05291 w_k1_MSFT_s106
                                 + 0.965161 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.03861 w_k1_ADM_s106
                                 + 1.05666 w_k1_AXP_s106
                                 + 1.12902 w_k1_DIS_s106
                                 + 1.14591 w_k1_FDX_s106
                                 + 0.960512 w_k1_IBM_s106
                                 + 0.950361 w_k1_INTC_s106
                                 + 1.05079 w_k1_KO_s106
                                 + 0.943392 w_k1_MAT_s106
                                 + 0.981587 w_k1_MSFT_s106
                                 + 1.04656 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.925705 w_k1_ADM_s106
                                 + 0.757536 w_k1_AXP_s106
                                 + 0.691111 w_k1_DIS_s106
                                 + 0.727391 w_k1_FDX_s106
                                 + 1.18362 w_k1_IBM_s106
                                 + 0.782985 w_k1_INTC_s106
                                 + 0.962663 w_k1_KO_s106
                                 + 1.44255 w_k1_MAT_s106
                                 + 0.789177 w_k1_MSFT_s106
                                 + 0.972215 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.14764 w_k1_ADM_s106
                                 + 1.34456 w_k1_AXP_s106
                                 + 1.60834 w_k1_DIS_s106
                                 + 1.35797 w_k1_FDX_s106
                                 + 1.28723 w_k1_IBM_s106
                                 + 1.0533 w_k1_INTC_s106 + 1.37694 w_k1_KO_s106
                                 + 1.15088 w_k1_MAT_s106
                                 + 1.1237 w_k1_MSFT_s106 + 1.25462 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.651089 w_k1_ADM_s106
                                 + 0.726604 w_k1_AXP_s106
                                 + 0.693893 w_k1_DIS_s106
                                 + 0.808585 w_k1_FDX_s106
                                 + 0.816376 w_k1_IBM_s106
                                 + 1.00734 w_k1_INTC_s106
                                 + 0.790702 w_k1_KO_s106
                                 + 0.826067 w_k1_MAT_s106
                                 + 0.866888 w_k1_MSFT_s106
                                 + 0.89172 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.26421 w_k1_ADM_s107
                                 + 1.45681 w_k1_AXP_s107 + 1.4035 w_k1_DIS_s107
                                 + 1.15606 w_k1_FDX_s107
                                 + 1.17743 w_k1_IBM_s107
                                 + 1.21115 w_k1_INTC_s107
                                 + 1.19203 w_k1_KO_s107 + 1.29967 w_k1_MAT_s107
                                 + 1.12519 w_k1_MSFT_s107
                                 + 1.14981 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.856173 w_k1_ADM_s107
                                 + 0.878965 w_k1_AXP_s107
                                 + 0.660572 w_k1_DIS_s107
                                 + 0.715889 w_k1_FDX_s107
                                 + 0.89156 w_k1_IBM_s107
                                 + 0.867531 w_k1_INTC_s107
                                 + 0.747085 w_k1_KO_s107
                                 + 1.07243 w_k1_MAT_s107
                                 + 0.763508 w_k1_MSFT_s107
                                 + 0.862625 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.860378 w_k1_ADM_s107
                                 + 0.88125 w_k1_AXP_s107
                                 + 0.889445 w_k1_DIS_s107
                                 + 0.839084 w_k1_FDX_s107
                                 + 1.07662 w_k1_IBM_s107
                                 + 1.0198 w_k1_INTC_s107
                                 + 0.973804 w_k1_KO_s107
                                 + 1.29454 w_k1_MAT_s107
                                 + 0.960263 w_k1_MSFT_s107
                                 + 0.944674 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.09873 w_k1_ADM_s107
                                 + 1.07602 w_k1_AXP_s107
                                 + 0.893093 w_k1_DIS_s107 + 1.068 w_k1_FDX_s107
                                 + 0.673198 w_k1_IBM_s107
                                 + 0.986097 w_k1_INTC_s107
                                 + 0.819094 w_k1_KO_s107
                                 + 0.688364 w_k1_MAT_s107
                                 + 0.927695 w_k1_MSFT_s107
                                 + 0.995215 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1246 w_k1_ADM_s107
                                 + 1.31954 w_k1_AXP_s107
                                 + 1.23334 w_k1_DIS_s107
                                 + 1.14691 w_k1_FDX_s107
                                 + 0.961665 w_k1_IBM_s107
                                 + 1.32756 w_k1_INTC_s107
                                 + 1.05745 w_k1_KO_s107 + 1.09658 w_k1_MAT_s107
                                 + 1.14417 w_k1_MSFT_s107
                                 + 1.05301 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.14527 w_k1_ADM_s107
                                 + 1.32831 w_k1_AXP_s107 + 1.0557 w_k1_DIS_s107
                                 + 1.09803 w_k1_FDX_s107
                                 + 0.782416 w_k1_IBM_s107
                                 + 1.04557 w_k1_INTC_s107
                                 + 0.95299 w_k1_KO_s107 + 1.17542 w_k1_MAT_s107
                                 + 1.08986 w_k1_MSFT_s107
                                 + 1.10087 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.22988 w_k1_ADM_s107
                                 + 1.54726 w_k1_AXP_s107
                                 + 1.28704 w_k1_DIS_s107
                                 + 1.07923 w_k1_FDX_s107
                                 + 1.45924 w_k1_IBM_s107
                                 + 1.05701 w_k1_INTC_s107
                                 + 1.23712 w_k1_KO_s107 + 1.89276 w_k1_MAT_s107
                                 + 1.04762 w_k1_MSFT_s107
                                 + 1.17221 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.942341 w_k1_ADM_s107
                                 + 0.705159 w_k1_AXP_s107
                                 + 0.715187 w_k1_DIS_s107
                                 + 0.778068 w_k1_FDX_s107
                                 + 0.682622 w_k1_IBM_s107
                                 + 0.829776 w_k1_INTC_s107
                                 + 0.730824 w_k1_KO_s107
                                 + 0.659507 w_k1_MAT_s107
                                 + 0.790979 w_k1_MSFT_s107
                                 + 0.821239 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.30481 w_k1_ADM_s110
                                 + 1.36077 w_k1_AXP_s110
                                 + 0.806709 w_k1_DIS_s110
                                 + 0.992496 w_k1_FDX_s110
                                 + 0.902831 w_k1_IBM_s110
                                 + 0.802554 w_k1_INTC_s110
                                 + 0.786156 w_k1_KO_s110
                                 + 0.995716 w_k1_MAT_s110
                                 + 0.761529 w_k1_MSFT_s110
                                 + 0.89923 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.07863 w_k1_ADM_s110
                                 + 0.836685 w_k1_AXP_s110
                                 + 0.60702 w_k1_DIS_s110
                                 + 0.807528 w_k1_FDX_s110
                                 + 0.953613 w_k1_IBM_s110
                                 + 0.741267 w_k1_INTC_s110
                                 + 0.829666 w_k1_KO_s110
                                 + 1.11216 w_k1_MAT_s110
                                 + 0.738688 w_k1_MSFT_s110
                                 + 0.891795 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.10988 w_k1_ADM_s110
                                 + 1.27253 w_k1_AXP_s110
                                 + 1.09351 w_k1_DIS_s110
                                 + 1.07244 w_k1_FDX_s110
                                 + 1.50389 w_k1_IBM_s110
                                 + 1.00977 w_k1_INTC_s110
                                 + 1.27056 w_k1_KO_s110 + 1.79959 w_k1_MAT_s110
                                 + 1.02383 w_k1_MSFT_s110
                                 + 1.23117 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.784652 w_k1_ADM_s110
                                 + 0.972394 w_k1_AXP_s110
                                 + 0.795392 w_k1_DIS_s110
                                 + 0.741158 w_k1_FDX_s110
                                 + 1.31977 w_k1_IBM_s110
                                 + 0.844915 w_k1_INTC_s110
                                 + 1.07575 w_k1_KO_s110 + 1.98147 w_k1_MAT_s110
                                 + 0.720998 w_k1_MSFT_s110
                                 + 0.930813 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.947775 w_k1_ADM_s110
                                 + 0.768121 w_k1_AXP_s110
                                 + 0.838421 w_k1_DIS_s110
                                 + 0.936557 w_k1_FDX_s110
                                 + 0.742955 w_k1_IBM_s110
                                 + 0.881755 w_k1_INTC_s110
                                 + 0.863732 w_k1_KO_s110
                                 + 0.841262 w_k1_MAT_s110
                                 + 0.85023 w_k1_MSFT_s110
                                 + 0.837409 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.89009 w_k1_ADM_s110
                                 + 1.64407 w_k1_AXP_s110
                                 + 1.60903 w_k1_DIS_s110
                                 + 1.46541 w_k1_FDX_s110
                                 + 1.09934 w_k1_IBM_s110
                                 + 1.06666 w_k1_INTC_s110
                                 + 1.10761 w_k1_KO_s110 + 1.04466 w_k1_MAT_s110
                                 + 1.22651 w_k1_MSFT_s110
                                 + 1.35544 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.23436 w_k1_ADM_s110
                                 + 1.14684 w_k1_AXP_s110
                                 + 1.13412 w_k1_DIS_s110
                                 + 1.20104 w_k1_FDX_s110
                                 + 0.856203 w_k1_IBM_s110
                                 + 0.984197 w_k1_INTC_s110
                                 + 1.11659 w_k1_KO_s110 + 1.0958 w_k1_MAT_s110
                                 + 1.01301 w_k1_MSFT_s110
                                 + 1.03525 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.96128 w_k1_ADM_s110
                                 + 1.0595 w_k1_AXP_s110 + 1.00499 w_k1_DIS_s110
                                 + 0.904136 w_k1_FDX_s110
                                 + 1.08608 w_k1_IBM_s110
                                 + 1.02378 w_k1_INTC_s110
                                 + 1.14993 w_k1_KO_s110 + 1.29991 w_k1_MAT_s110
                                 + 1.08121 w_k1_MSFT_s110 + 1.1117 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.787475 w_k1_ADM_s111
                                 + 1.39844 w_k1_AXP_s111
                                 + 0.857168 w_k1_DIS_s111
                                 + 1.21344 w_k1_FDX_s111
                                 + 0.764564 w_k1_IBM_s111
                                 + 1.0813 w_k1_INTC_s111 + 0.78869 w_k1_KO_s111
                                 + 1.65823 w_k1_MAT_s111
                                 + 1.04917 w_k1_MSFT_s111
                                 + 1.07641 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.77756 w_k1_ADM_s111
                                 + 0.632821 w_k1_AXP_s111
                                 + 0.610393 w_k1_DIS_s111
                                 + 0.768893 w_k1_FDX_s111
                                 + 0.720939 w_k1_IBM_s111
                                 + 0.920443 w_k1_INTC_s111
                                 + 0.736363 w_k1_KO_s111
                                 + 0.770658 w_k1_MAT_s111
                                 + 0.797637 w_k1_MSFT_s111
                                 + 0.811903 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.27663 w_k1_ADM_s111
                                 + 1.51802 w_k1_AXP_s111
                                 + 1.68432 w_k1_DIS_s111
                                 + 1.31754 w_k1_FDX_s111
                                 + 1.52148 w_k1_IBM_s111
                                 + 1.34068 w_k1_INTC_s111
                                 + 1.38626 w_k1_KO_s111 + 1.75786 w_k1_MAT_s111
                                 + 1.18958 w_k1_MSFT_s111
                                 + 1.28589 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.759243 w_k1_ADM_s111
                                 + 0.578345 w_k1_AXP_s111
                                 + 0.537489 w_k1_DIS_s111
                                 + 0.701875 w_k1_FDX_s111
                                 + 0.861963 w_k1_IBM_s111
                                 + 0.629836 w_k1_INTC_s111
                                 + 0.781242 w_k1_KO_s111
                                 + 0.852118 w_k1_MAT_s111
                                 + 0.650999 w_k1_MSFT_s111
                                 + 0.80965 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.23207 w_k1_ADM_s111
                                 + 1.07098 w_k1_AXP_s111
                                 + 1.07677 w_k1_DIS_s111
                                 + 1.06893 w_k1_FDX_s111 + 1.2839 w_k1_IBM_s111
                                 + 0.918097 w_k1_INTC_s111
                                 + 1.06918 w_k1_KO_s111 + 1.19457 w_k1_MAT_s111
                                 + 0.860008 w_k1_MSFT_s111
                                 + 1.18215 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.932907 w_k1_ADM_s111
                                 + 0.862971 w_k1_AXP_s111
                                 + 1.07998 w_k1_DIS_s111
                                 + 1.11489 w_k1_FDX_s111
                                 + 0.803631 w_k1_IBM_s111
                                 + 0.84939 w_k1_INTC_s111
                                 + 0.907381 w_k1_KO_s111
                                 + 1.07967 w_k1_MAT_s111
                                 + 0.990358 w_k1_MSFT_s111
                                 + 1.00578 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.22871 w_k1_ADM_s111
                                 + 0.98037 w_k1_AXP_s111
                                 + 1.38795 w_k1_DIS_s111
                                 + 1.12254 w_k1_FDX_s111
                                 + 1.04555 w_k1_IBM_s111
                                 + 0.903934 w_k1_INTC_s111
                                 + 1.20256 w_k1_KO_s111 + 1.28939 w_k1_MAT_s111
                                 + 1.04779 w_k1_MSFT_s111
                                 + 1.13507 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.841364 w_k1_ADM_s111
                                 + 0.713513 w_k1_AXP_s111
                                 + 0.844224 w_k1_DIS_s111
                                 + 0.861996 w_k1_FDX_s111
                                 + 1.06079 w_k1_IBM_s111
                                 + 0.858925 w_k1_INTC_s111
                                 + 0.960077 w_k1_KO_s111
                                 + 1.08849 w_k1_MAT_s111
                                 + 0.835196 w_k1_MSFT_s111
                                 + 0.892501 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.35575 w_k1_ADM_s112
                                 + 1.23972 w_k1_AXP_s112
                                 + 1.73789 w_k1_DIS_s112
                                 + 1.33501 w_k1_FDX_s112
                                 + 1.15661 w_k1_IBM_s112
                                 + 1.19687 w_k1_INTC_s112
                                 + 1.22481 w_k1_KO_s112
                                 + 0.990528 w_k1_MAT_s112
                                 + 1.42077 w_k1_MSFT_s112
                                 + 1.24608 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.934417 w_k1_ADM_s112
                                 + 1.12447 w_k1_AXP_s112
                                 + 0.856745 w_k1_DIS_s112
                                 + 0.876254 w_k1_FDX_s112
                                 + 1.35421 w_k1_IBM_s112
                                 + 0.776277 w_k1_INTC_s112
                                 + 0.872041 w_k1_KO_s112
                                 + 1.24617 w_k1_MAT_s112
                                 + 0.708339 w_k1_MSFT_s112
                                 + 0.967001 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.18654 w_k1_ADM_s112
                                 + 0.992568 w_k1_AXP_s112
                                 + 1.14567 w_k1_DIS_s112
                                 + 0.937437 w_k1_FDX_s112
                                 + 1.29723 w_k1_IBM_s112
                                 + 0.854226 w_k1_INTC_s112
                                 + 1.33963 w_k1_KO_s112 + 1.17459 w_k1_MAT_s112
                                 + 0.957558 w_k1_MSFT_s112
                                 + 1.07124 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.946016 w_k1_ADM_s112
                                 + 0.897921 w_k1_AXP_s112
                                 + 0.857801 w_k1_DIS_s112
                                 + 0.87404 w_k1_FDX_s112
                                 + 0.881177 w_k1_IBM_s112
                                 + 0.734012 w_k1_INTC_s112
                                 + 0.79999 w_k1_KO_s112
                                 + 0.939188 w_k1_MAT_s112
                                 + 0.875131 w_k1_MSFT_s112
                                 + 0.982582 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.43412 w_k1_ADM_s112
                                 + 1.19485 w_k1_AXP_s112
                                 + 1.28275 w_k1_DIS_s112
                                 + 1.16641 w_k1_FDX_s112
                                 + 1.33802 w_k1_IBM_s112
                                 + 0.95941 w_k1_INTC_s112
                                 + 1.26184 w_k1_KO_s112 + 1.45991 w_k1_MAT_s112
                                 + 1.18755 w_k1_MSFT_s112
                                 + 1.19971 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.779453 w_k1_ADM_s112
                                 + 0.567856 w_k1_AXP_s112
                                 + 0.608346 w_k1_DIS_s112
                                 + 0.746555 w_k1_FDX_s112
                                 + 0.692694 w_k1_IBM_s112
                                 + 0.609538 w_k1_INTC_s112
                                 + 0.648166 w_k1_KO_s112
                                 + 0.755613 w_k1_MAT_s112
                                 + 0.756181 w_k1_MSFT_s112
                                 + 0.888127 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.5176 w_k1_ADM_s112
                                 + 1.50764 w_k1_AXP_s112
                                 + 1.14562 w_k1_DIS_s112
                                 + 1.25646 w_k1_FDX_s112
                                 + 1.02886 w_k1_IBM_s112
                                 + 1.01428 w_k1_INTC_s112
                                 + 0.932488 w_k1_KO_s112
                                 + 0.808376 w_k1_MAT_s112
                                 + 1.05423 w_k1_MSFT_s112
                                 + 1.23394 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.14173 w_k1_ADM_s112
                                 + 0.991873 w_k1_AXP_s112
                                 + 0.991218 w_k1_DIS_s112
                                 + 1.07878 w_k1_FDX_s112
                                 + 0.946101 w_k1_IBM_s112
                                 + 1.14256 w_k1_INTC_s112
                                 + 1.00164 w_k1_KO_s112
                                 + 0.621878 w_k1_MAT_s112
                                 + 0.949123 w_k1_MSFT_s112
                                 + 1.04639 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.18372 w_k1_ADM_s113
                                 + 1.21828 w_k1_AXP_s113
                                 + 1.40306 w_k1_DIS_s113
                                 + 1.21785 w_k1_FDX_s113
                                 + 1.03882 w_k1_IBM_s113
                                 + 0.976841 w_k1_INTC_s113
                                 + 1.28418 w_k1_KO_s113 + 1.05986 w_k1_MAT_s113
                                 + 1.10615 w_k1_MSFT_s113
                                 + 1.06984 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.07916 w_k1_ADM_s113
                                 + 1.21763 w_k1_AXP_s113
                                 + 1.07035 w_k1_DIS_s113
                                 + 1.21366 w_k1_FDX_s113 + 0.7468 w_k1_IBM_s113
                                 + 1.12535 w_k1_INTC_s113
                                 + 0.90181 w_k1_KO_s113 + 1.21112 w_k1_MAT_s113
                                 + 1.09241 w_k1_MSFT_s113
                                 + 1.07487 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.1922 w_k1_ADM_s113
                                 + 0.961348 w_k1_AXP_s113
                                 + 0.977641 w_k1_DIS_s113
                                 + 1.01998 w_k1_FDX_s113
                                 + 0.876807 w_k1_IBM_s113
                                 + 0.930022 w_k1_INTC_s113
                                 + 0.949563 w_k1_KO_s113
                                 + 0.566259 w_k1_MAT_s113
                                 + 0.996396 w_k1_MSFT_s113
                                 + 0.958025 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.911943 w_k1_ADM_s113
                                 + 0.868694 w_k1_AXP_s113
                                 + 0.772619 w_k1_DIS_s113
                                 + 0.933672 w_k1_FDX_s113
                                 + 0.621134 w_k1_IBM_s113
                                 + 0.953572 w_k1_INTC_s113
                                 + 0.924159 w_k1_KO_s113
                                 + 0.714563 w_k1_MAT_s113
                                 + 1.01579 w_k1_MSFT_s113
                                 + 0.954776 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.56882 w_k1_ADM_s113
                                 + 1.51783 w_k1_AXP_s113
                                 + 1.43995 w_k1_DIS_s113
                                 + 1.39864 w_k1_FDX_s113
                                 + 1.15593 w_k1_IBM_s113
                                 + 1.35786 w_k1_INTC_s113
                                 + 1.32917 w_k1_KO_s113 + 1.07136 w_k1_MAT_s113
                                 + 1.31258 w_k1_MSFT_s113
                                 + 1.22939 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.957193 w_k1_ADM_s113
                                 + 1.32168 w_k1_AXP_s113
                                 + 1.17914 w_k1_DIS_s113
                                 + 1.08195 w_k1_FDX_s113
                                 + 1.22611 w_k1_IBM_s113
                                 + 1.0455 w_k1_INTC_s113 + 1.18897 w_k1_KO_s113
                                 + 1.27621 w_k1_MAT_s113
                                 + 1.02825 w_k1_MSFT_s113
                                 + 1.00095 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.863837 w_k1_ADM_s113
                                 + 0.730299 w_k1_AXP_s113
                                 + 0.734843 w_k1_DIS_s113
                                 + 0.795437 w_k1_FDX_s113
                                 + 0.833813 w_k1_IBM_s113
                                 + 0.86464 w_k1_INTC_s113
                                 + 0.856345 w_k1_KO_s113
                                 + 0.817182 w_k1_MAT_s113
                                 + 0.897647 w_k1_MSFT_s113
                                 + 1.00083 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.885301 w_k1_ADM_s113
                                 + 1.13036 w_k1_AXP_s113
                                 + 0.862784 w_k1_DIS_s113
                                 + 0.834292 w_k1_FDX_s113
                                 + 1.25136 w_k1_IBM_s113
                                 + 1.01444 w_k1_INTC_s113
                                 + 1.10093 w_k1_KO_s113 + 1.1393 w_k1_MAT_s113
                                 + 0.891579 w_k1_MSFT_s113
                                 + 1.00538 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.930342 w_k1_ADM_s114
                                 + 0.946024 w_k1_AXP_s114
                                 + 0.669013 w_k1_DIS_s114
                                 + 0.905332 w_k1_FDX_s114
                                 + 0.765238 w_k1_IBM_s114
                                 + 0.938463 w_k1_INTC_s114
                                 + 0.864496 w_k1_KO_s114
                                 + 0.764734 w_k1_MAT_s114
                                 + 0.861725 w_k1_MSFT_s114
                                 + 0.933905 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.02244 w_k1_ADM_s114
                                 + 1.55897 w_k1_AXP_s114
                                 + 1.28966 w_k1_DIS_s114
                                 + 1.21769 w_k1_FDX_s114
                                 + 1.11492 w_k1_IBM_s114
                                 + 1.37534 w_k1_INTC_s114
                                 + 1.45659 w_k1_KO_s114 + 1.11632 w_k1_MAT_s114
                                 + 1.16086 w_k1_MSFT_s114
                                 + 0.957798 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.806154 w_k1_ADM_s114
                                 + 0.985082 w_k1_AXP_s114
                                 + 1.37568 w_k1_DIS_s114
                                 + 1.10461 w_k1_FDX_s114
                                 + 1.00188 w_k1_IBM_s114
                                 + 1.26404 w_k1_INTC_s114
                                 + 1.21886 w_k1_KO_s114 + 1.15205 w_k1_MAT_s114
                                 + 1.33958 w_k1_MSFT_s114
                                 + 1.08123 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.595764 w_k1_ADM_s114
                                 + 0.551493 w_k1_AXP_s114
                                 + 0.535336 w_k1_DIS_s114
                                 + 0.640265 w_k1_FDX_s114
                                 + 0.811846 w_k1_IBM_s114
                                 + 0.826195 w_k1_INTC_s114
                                 + 0.992343 w_k1_KO_s114
                                 + 0.87512 w_k1_MAT_s114
                                 + 0.800986 w_k1_MSFT_s114
                                 + 0.882672 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.94333 w_k1_ADM_s114
                                 + 1.01937 w_k1_AXP_s114 + 1.0195 w_k1_DIS_s114
                                 + 0.967671 w_k1_FDX_s114
                                 + 1.17249 w_k1_IBM_s114
                                 + 1.01403 w_k1_INTC_s114
                                 + 1.34836 w_k1_KO_s114 + 1.6218 w_k1_MAT_s114
                                 + 1.10816 w_k1_MSFT_s114
                                 + 0.953121 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.692516 w_k1_ADM_s114
                                 + 0.581227 w_k1_AXP_s114
                                 + 0.844702 w_k1_DIS_s114
                                 + 0.821142 w_k1_FDX_s114
                                 + 0.713818 w_k1_IBM_s114
                                 + 1.0446 w_k1_INTC_s114
                                 + 0.983309 w_k1_KO_s114
                                 + 0.365883 w_k1_MAT_s114
                                 + 0.96165 w_k1_MSFT_s114
                                 + 0.839082 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.16663 w_k1_ADM_s114
                                 + 0.836303 w_k1_AXP_s114
                                 + 1.12878 w_k1_DIS_s114
                                 + 1.11224 w_k1_FDX_s114
                                 + 0.641392 w_k1_IBM_s114
                                 + 1.01686 w_k1_INTC_s114
                                 + 0.935765 w_k1_KO_s114
                                 + 0.705378 w_k1_MAT_s114
                                 + 1.22276 w_k1_MSFT_s114
                                 + 0.928411 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.940189 w_k1_ADM_s114
                                 + 1.11102 w_k1_AXP_s114
                                 + 1.21715 w_k1_DIS_s114
                                 + 1.02315 w_k1_FDX_s114
                                 + 1.78086 w_k1_IBM_s114
                                 + 1.12245 w_k1_INTC_s114
                                 + 1.76044 w_k1_KO_s114 + 1.6471 w_k1_MAT_s114
                                 + 1.33149 w_k1_MSFT_s114
                                 + 1.13885 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.09796 w_k1_ADM_s115
                                 + 0.920715 w_k1_AXP_s115
                                 + 1.29697 w_k1_DIS_s115
                                 + 1.13789 w_k1_FDX_s115
                                 + 0.94163 w_k1_IBM_s115
                                 + 1.30783 w_k1_INTC_s115
                                 + 1.07684 w_k1_KO_s115
                                 + 0.800518 w_k1_MAT_s115
                                 + 1.21866 w_k1_MSFT_s115
                                 + 1.25622 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.980556 w_k1_ADM_s115
                                 + 1.21932 w_k1_AXP_s115
                                 + 1.05831 w_k1_DIS_s115
                                 + 1.02591 w_k1_FDX_s115
                                 + 1.52642 w_k1_IBM_s115
                                 + 1.12058 w_k1_INTC_s115 + 1.2636 w_k1_KO_s115
                                 + 1.65901 w_k1_MAT_s115
                                 + 1.01217 w_k1_MSFT_s115
                                 + 1.21771 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.97882 w_k1_ADM_s115
                                 + 1.14362 w_k1_AXP_s115
                                 + 0.945591 w_k1_DIS_s115
                                 + 1.07313 w_k1_FDX_s115
                                 + 1.00103 w_k1_IBM_s115
                                 + 1.09013 w_k1_INTC_s115
                                 + 1.09774 w_k1_KO_s115 + 1.33733 w_k1_MAT_s115
                                 + 1.02534 w_k1_MSFT_s115
                                 + 1.19632 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.759131 w_k1_ADM_s115
                                 + 0.695565 w_k1_AXP_s115
                                 + 0.75314 w_k1_DIS_s115
                                 + 0.803711 w_k1_FDX_s115
                                 + 0.896626 w_k1_IBM_s115
                                 + 0.929584 w_k1_INTC_s115
                                 + 0.802298 w_k1_KO_s115
                                 + 0.75982 w_k1_MAT_s115
                                 + 0.865903 w_k1_MSFT_s115
                                 + 0.933455 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.03445 w_k1_ADM_s115
                                 + 0.78347 w_k1_AXP_s115
                                 + 0.745813 w_k1_DIS_s115
                                 + 0.888783 w_k1_FDX_s115
                                 + 0.619745 w_k1_IBM_s115
                                 + 0.973842 w_k1_INTC_s115
                                 + 0.871048 w_k1_KO_s115
                                 + 0.549835 w_k1_MAT_s115
                                 + 0.890042 w_k1_MSFT_s115
                                 + 1.07139 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.966488 w_k1_ADM_s115
                                 + 1.00348 w_k1_AXP_s115 + 1.3119 w_k1_DIS_s115
                                 + 1.24458 w_k1_FDX_s115 + 1.345 w_k1_IBM_s115
                                 + 1.07571 w_k1_INTC_s115
                                 + 1.31683 w_k1_KO_s115 + 1.55485 w_k1_MAT_s115
                                 + 1.16219 w_k1_MSFT_s115
                                 + 1.30779 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.68168 w_k1_ADM_s115
                                 + 1.39763 w_k1_AXP_s115
                                 + 1.71796 w_k1_DIS_s115
                                 + 1.57869 w_k1_FDX_s115 + 1.1857 w_k1_IBM_s115
                                 + 1.50629 w_k1_INTC_s115
                                 + 1.44038 w_k1_KO_s115 + 1.63775 w_k1_MAT_s115
                                 + 1.51049 w_k1_MSFT_s115
                                 + 1.36992 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12093 w_k1_ADM_s115
                                 + 0.783353 w_k1_AXP_s115
                                 + 0.76624 w_k1_DIS_s115
                                 + 0.816463 w_k1_FDX_s115
                                 + 1.21102 w_k1_IBM_s115
                                 + 0.75484 w_k1_INTC_s115
                                 + 0.980264 w_k1_KO_s115 + 1.2494 w_k1_MAT_s115
                                 + 0.860704 w_k1_MSFT_s115
                                 + 1.1485 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.910845 w_k1_ADM_s116
                                 + 1.18387 w_k1_AXP_s116
                                 + 1.11905 w_k1_DIS_s116
                                 + 1.09833 w_k1_FDX_s116
                                 + 0.978227 w_k1_IBM_s116
                                 + 1.06343 w_k1_INTC_s116
                                 + 0.985195 w_k1_KO_s116
                                 + 1.16554 w_k1_MAT_s116
                                 + 0.998277 w_k1_MSFT_s116
                                 + 1.0183 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.779681 w_k1_ADM_s116
                                 + 0.923233 w_k1_AXP_s116
                                 + 0.762951 w_k1_DIS_s116
                                 + 0.784295 w_k1_FDX_s116
                                 + 1.03063 w_k1_IBM_s116
                                 + 0.899679 w_k1_INTC_s116
                                 + 0.908533 w_k1_KO_s116
                                 + 1.05367 w_k1_MAT_s116
                                 + 0.838775 w_k1_MSFT_s116
                                 + 0.880155 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.87712 w_k1_ADM_s116
                                 + 1.16635 w_k1_AXP_s116
                                 + 1.02648 w_k1_DIS_s116
                                 + 0.943737 w_k1_FDX_s116
                                 + 1.3527 w_k1_IBM_s116 + 1.1492 w_k1_INTC_s116
                                 + 1.13044 w_k1_KO_s116 + 1.32098 w_k1_MAT_s116
                                 + 0.967 w_k1_MSFT_s116 + 0.949825 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.829091 w_k1_ADM_s116
                                 + 0.794004 w_k1_AXP_s116
                                 + 0.756874 w_k1_DIS_s116
                                 + 0.909109 w_k1_FDX_s116
                                 + 0.650449 w_k1_IBM_s116
                                 + 1.10079 w_k1_INTC_s116
                                 + 0.722292 w_k1_KO_s116
                                 + 0.556837 w_k1_MAT_s116
                                 + 0.924922 w_k1_MSFT_s116
                                 + 0.897559 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.33627 w_k1_ADM_s116
                                 + 1.45563 w_k1_AXP_s116 + 1.4699 w_k1_DIS_s116
                                 + 1.26249 w_k1_FDX_s116
                                 + 1.28406 w_k1_IBM_s116
                                 + 1.21271 w_k1_INTC_s116
                                 + 1.39468 w_k1_KO_s116 + 1.90989 w_k1_MAT_s116
                                 + 1.11264 w_k1_MSFT_s116
                                 + 1.20052 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.894117 w_k1_ADM_s116
                                 + 0.903195 w_k1_AXP_s116
                                 + 1.05784 w_k1_DIS_s116
                                 + 0.965525 w_k1_FDX_s116
                                 + 0.79719 w_k1_IBM_s116
                                 + 1.03236 w_k1_INTC_s116 + 1.0575 w_k1_KO_s116
                                 + 1.13623 w_k1_MAT_s116
                                 + 1.07001 w_k1_MSFT_s116
                                 + 1.02354 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.02416 w_k1_ADM_s116
                                 + 1.04252 w_k1_AXP_s116
                                 + 1.41622 w_k1_DIS_s116
                                 + 1.09783 w_k1_FDX_s116
                                 + 0.98958 w_k1_IBM_s116
                                 + 1.13592 w_k1_INTC_s116
                                 + 1.24476 w_k1_KO_s116
                                 + 0.762711 w_k1_MAT_s116
                                 + 1.13254 w_k1_MSFT_s116
                                 + 1.04887 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.12022 w_k1_ADM_s116
                                 + 1.0335 w_k1_AXP_s116 + 1.19862 w_k1_DIS_s116
                                 + 1.05148 w_k1_FDX_s116
                                 + 1.59259 w_k1_IBM_s116
                                 + 0.91647 w_k1_INTC_s116
                                 + 1.40812 w_k1_KO_s116 + 1.83456 w_k1_MAT_s116
                                 + 1.04679 w_k1_MSFT_s116
                                 + 1.10131 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.927187 w_k1_ADM_s117
                                 + 0.794989 w_k1_AXP_s117
                                 + 1.11319 w_k1_DIS_s117
                                 + 0.917143 w_k1_FDX_s117
                                 + 1.16661 w_k1_IBM_s117
                                 + 1.15882 w_k1_INTC_s117
                                 + 1.11849 w_k1_KO_s117
                                 + 0.938294 w_k1_MAT_s117
                                 + 1.03351 w_k1_MSFT_s117
                                 + 0.921293 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.798701 w_k1_ADM_s117
                                 + 0.862543 w_k1_AXP_s117
                                 + 0.936954 w_k1_DIS_s117
                                 + 0.928353 w_k1_FDX_s117
                                 + 1.33347 w_k1_IBM_s117
                                 + 0.96099 w_k1_INTC_s117
                                 + 1.19208 w_k1_KO_s117 + 1.47101 w_k1_MAT_s117
                                 + 0.994013 w_k1_MSFT_s117
                                 + 0.970825 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.41855 w_k1_ADM_s117
                                 + 1.30107 w_k1_AXP_s117
                                 + 1.59238 w_k1_DIS_s117
                                 + 1.37296 w_k1_FDX_s117
                                 + 1.21998 w_k1_IBM_s117
                                 + 1.20641 w_k1_INTC_s117
                                 + 1.25939 w_k1_KO_s117 + 1.3218 w_k1_MAT_s117
                                 + 1.35729 w_k1_MSFT_s117
                                 + 1.10553 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.86876 w_k1_ADM_s117
                                 + 1.04131 w_k1_AXP_s117
                                 + 1.16292 w_k1_DIS_s117
                                 + 1.18564 w_k1_FDX_s117
                                 + 0.913026 w_k1_IBM_s117
                                 + 1.16827 w_k1_INTC_s117
                                 + 1.07991 w_k1_KO_s117
                                 + 0.942517 w_k1_MAT_s117
                                 + 1.0941 w_k1_MSFT_s117
                                 + 0.900157 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.14686 w_k1_ADM_s117
                                 + 0.92534 w_k1_AXP_s117
                                 + 0.857524 w_k1_DIS_s117
                                 + 0.979521 w_k1_FDX_s117
                                 + 0.998208 w_k1_IBM_s117
                                 + 0.815354 w_k1_INTC_s117
                                 + 0.874138 w_k1_KO_s117
                                 + 0.819136 w_k1_MAT_s117
                                 + 1.04649 w_k1_MSFT_s117
                                 + 0.961275 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.948511 w_k1_ADM_s117
                                 + 0.783128 w_k1_AXP_s117
                                 + 0.784181 w_k1_DIS_s117
                                 + 0.872864 w_k1_FDX_s117
                                 + 0.732505 w_k1_IBM_s117
                                 + 0.879212 w_k1_INTC_s117
                                 + 0.929849 w_k1_KO_s117
                                 + 0.930461 w_k1_MAT_s117
                                 + 1.03409 w_k1_MSFT_s117
                                 + 0.986464 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.694233 w_k1_ADM_s117
                                 + 0.627688 w_k1_AXP_s117
                                 + 0.680249 w_k1_DIS_s117
                                 + 0.69738 w_k1_FDX_s117
                                 + 0.782761 w_k1_IBM_s117
                                 + 0.736355 w_k1_INTC_s117
                                 + 0.685206 w_k1_KO_s117
                                 + 0.877184 w_k1_MAT_s117
                                 + 0.858403 w_k1_MSFT_s117
                                 + 0.724741 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.890462 w_k1_ADM_s117
                                 + 0.961508 w_k1_AXP_s117
                                 + 0.900606 w_k1_DIS_s117
                                 + 0.919399 w_k1_FDX_s117
                                 + 1.05257 w_k1_IBM_s117
                                 + 1.00043 w_k1_INTC_s117
                                 + 0.88268 w_k1_KO_s117 + 0.87291 w_k1_MAT_s117
                                 + 0.872151 w_k1_MSFT_s117
                                 + 0.854668 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0343828170461981 w_k1_ADM_s000
                                 + 0.0277866138061324 w_k1_ADM_s001
                                 + 0.0375205360100565 w_k1_ADM_s002
                                 + 0.0225570351441989 w_k1_ADM_s003
                                 + 0.0549584642178813 w_k1_ADM_s004
                                 + 0.0207245622369353 w_k1_ADM_s005
                                 + 0.0278895909963984 w_k1_ADM_s006
                                 + 0.0267036047601467 w_k1_ADM_s007
                                 + 0.0127822207148954 w_k1_ADM_s010
                                 + 0.0386250650757542 w_k1_ADM_s011
                                 + 0.0234620660384648 w_k1_ADM_s012
                                 + 0.00941508235343231 w_k1_ADM_s013
                                 + 0.0366455249076729 w_k1_ADM_s014
                                 + 0.0144101044637939 w_k1_ADM_s015
                                 + 0.0171727522330382 w_k1_ADM_s016
                                 + 0.0135034079617355 w_k1_ADM_s017
                                 + 0.0938344629209951 w_k1_ADM_s100
                                 + 0.111688840106908 w_k1_ADM_s101
                                 + 0.0876780884398646 w_k1_ADM_s102
                                 + 0.0331499299091181 w_k1_ADM_s103
                                 + 0.0310549754549461 w_k1_ADM_s104
                                 + 0.029602167860293 w_k1_ADM_s105
                                 + 0.029338774959762 w_k1_ADM_s106
                                 + 0.0359935142192043 w_k1_ADM_s107
                                 + 0.0163615156207743 w_k1_ADM_s110
                                 + 0.0129292357287028 w_k1_ADM_s111
                                 + 0.0116777738492501 w_k1_ADM_s112
                                 + 0.0109824810669589 w_k1_ADM_s113
                                 + 0.020705984175792 w_k1_ADM_s114
                                 + 0.0222489231624468 w_k1_ADM_s115
                                 + 0.0238674272159097 w_k1_ADM_s116
                                 + 0.0084412882968521 w_k1_ADM_s117
                                 + 0.0322146428402257 w_k1_AXP_s000
                                 + 0.0281695074244683 w_k1_AXP_s001
                                 + 0.0394983423501273 w_k1_AXP_s002
                                 + 0.0241593845948674 w_k1_AXP_s003
                                 + 0.0548740526906897 w_k1_AXP_s004
                                 + 0.0214245794854488 w_k1_AXP_s005
                                 + 0.0305792967362305 w_k1_AXP_s006
                                 + 0.0274608640317149 w_k1_AXP_s007
                                 + 0.0120209069291921 w_k1_AXP_s010
                                 + 0.0434087672879301 w_k1_AXP_s011
                                 + 0.0242413504721872 w_k1_AXP_s012
                                 + 0.00939583626177384 w_k1_AXP_s013
                                 + 0.0371780601496772 w_k1_AXP_s014
                                 + 0.0144371994144651 w_k1_AXP_s015
                                 + 0.0144604014239771 w_k1_AXP_s016
                                 + 0.0150352872587371 w_k1_AXP_s017
                                 + 0.0920846736858357 w_k1_AXP_s100
                                 + 0.113996962412992 w_k1_AXP_s101
                                 + 0.0899565847071686 w_k1_AXP_s102
                                 + 0.0300008661988457 w_k1_AXP_s103
                                 + 0.0321752441918882 w_k1_AXP_s104
                                 + 0.0291806562500389 w_k1_AXP_s105
                                 + 0.0281957012364098 w_k1_AXP_s106
                                 + 0.0392191686641226 w_k1_AXP_s107
                                 + 0.0155971622479738 w_k1_AXP_s110
                                 + 0.0121538710382481 w_k1_AXP_s111
                                 + 0.0106536779766175 w_k1_AXP_s112
                                 + 0.011181435193846 w_k1_AXP_s113
                                 + 0.0228911469285241 w_k1_AXP_s114
                                 + 0.0211225577832341 w_k1_AXP_s115
                                 + 0.0259657469513777 w_k1_AXP_s116
                                 + 0.00822378189864794 w_k1_AXP_s117
                                 + 0.03075791691263 w_k1_DIS_s000
                                 + 0.0284808838608867 w_k1_DIS_s001
                                 + 0.038535583879436 w_k1_DIS_s002
                                 + 0.0263485204566767 w_k1_DIS_s003
                                 + 0.0538667206232577 w_k1_DIS_s004
                                 + 0.024604054700191 w_k1_DIS_s005
                                 + 0.0282979996818313 w_k1_DIS_s006
                                 + 0.0282503131306156 w_k1_DIS_s007
                                 + 0.0134348986760364 w_k1_DIS_s010
                                 + 0.0432483704575598 w_k1_DIS_s011
                                 + 0.0238651790308862 w_k1_DIS_s012
                                 + 0.00940927666619631 w_k1_DIS_s013
                                 + 0.0419873894752057 w_k1_DIS_s014
                                 + 0.015643203785029 w_k1_DIS_s015
                                 + 0.0151533203440635 w_k1_DIS_s016
                                 + 0.0153287271576831 w_k1_DIS_s017
                                 + 0.100995298302097 w_k1_DIS_s100
                                 + 0.116167202751494 w_k1_DIS_s101
                                 + 0.0895044208447101 w_k1_DIS_s102
                                 + 0.0317720630117222 w_k1_DIS_s103
                                 + 0.03369284655339 w_k1_DIS_s104
                                 + 0.0325136553399941 w_k1_DIS_s105
                                 + 0.0320482243677234 w_k1_DIS_s106
                                 + 0.0351196826327843 w_k1_DIS_s107
                                 + 0.0138550732428294 w_k1_DIS_s110
                                 + 0.0137130244151029 w_k1_DIS_s111
                                 + 0.0105584758357215 w_k1_DIS_s112
                                 + 0.0106100590442628 w_k1_DIS_s113
                                 + 0.0238889895145598 w_k1_DIS_s114
                                 + 0.0222755834641356 w_k1_DIS_s115
                                 + 0.0264876532790851 w_k1_DIS_s116
                                 + 0.00899536164973272 w_k1_DIS_s117
                                 + 0.0319401977034414 w_k1_FDX_s000
                                 + 0.0274066159776667 w_k1_FDX_s001
                                 + 0.0382128125931642 w_k1_FDX_s002
                                 + 0.0263295106309911 w_k1_FDX_s003
                                 + 0.0543318901413869 w_k1_FDX_s004
                                 + 0.0238359646589648 w_k1_FDX_s005
                                 + 0.0299096065515402 w_k1_FDX_s006
                                 + 0.0270640203592411 w_k1_FDX_s007
                                 + 0.0120738468041409 w_k1_FDX_s010
                                 + 0.0406153558779145 w_k1_FDX_s011
                                 + 0.0223933002108718 w_k1_FDX_s012
                                 + 0.00872429684286205 w_k1_FDX_s013
                                 + 0.0385048305463957 w_k1_FDX_s014
                                 + 0.0158511160569674 w_k1_FDX_s015
                                 + 0.0143683101962717 w_k1_FDX_s016
                                 + 0.0141092077663496 w_k1_FDX_s017
                                 + 0.0993035683913662 w_k1_FDX_s100
                                 + 0.113418573204705 w_k1_FDX_s101
                                 + 0.0898890705682046 w_k1_FDX_s102
                                 + 0.0309865459674133 w_k1_FDX_s103
                                 + 0.0328741146375734 w_k1_FDX_s104
                                 + 0.0315563955608902 w_k1_FDX_s105
                                 + 0.0312373657700151 w_k1_FDX_s106
                                 + 0.0331601604194316 w_k1_FDX_s107
                                 + 0.0144715846494291 w_k1_FDX_s110
                                 + 0.0132839258131081 w_k1_FDX_s111
                                 + 0.0104292869769948 w_k1_FDX_s112
                                 + 0.0106854552956561 w_k1_FDX_s113
                                 + 0.0226670557902061 w_k1_FDX_s114
                                 + 0.0228068201927761 w_k1_FDX_s115
                                 + 0.0251813888150019 w_k1_FDX_s116
                                 + 0.00890004197890432 w_k1_FDX_s117
                                 + 0.031757055702466 w_k1_IBM_s000
                                 + 0.0274541944978982 w_k1_IBM_s001
                                 + 0.0402274742355909 w_k1_IBM_s002
                                 + 0.0260784230660184 w_k1_IBM_s003
                                 + 0.0512888605746412 w_k1_IBM_s004
                                 + 0.0221499948524379 w_k1_IBM_s005
                                 + 0.0310243481562151 w_k1_IBM_s006
                                 + 0.028624765021385 w_k1_IBM_s007
                                 + 0.0134829286994373 w_k1_IBM_s010
                                 + 0.0410747775914369 w_k1_IBM_s011
                                 + 0.0246830018783581 w_k1_IBM_s012
                                 + 0.00838254928920339 w_k1_IBM_s013
                                 + 0.0425435622055357 w_k1_IBM_s014
                                 + 0.0158665775363418 w_k1_IBM_s015
                                 + 0.0143061944043188 w_k1_IBM_s016
                                 + 0.0153113986105734 w_k1_IBM_s017
                                 + 0.10111181253292 w_k1_IBM_s100
                                 + 0.113668870609944 w_k1_IBM_s101
                                 + 0.0890287996906067 w_k1_IBM_s102
                                 + 0.0315520710505704 w_k1_IBM_s103
                                 + 0.0333013880367363 w_k1_IBM_s104
                                 + 0.0321435331025981 w_k1_IBM_s105
                                 + 0.0305730519925924 w_k1_IBM_s106
                                 + 0.0340630351192996 w_k1_IBM_s107
                                 + 0.0147241746761858 w_k1_IBM_s110
                                 + 0.0131449649184835 w_k1_IBM_s111
                                 + 0.0107272055080653 w_k1_IBM_s112
                                 + 0.010062798292169 w_k1_IBM_s113
                                 + 0.0233007179505096 w_k1_IBM_s114
                                 + 0.022962524792222 w_k1_IBM_s115
                                 + 0.0255551974379592 w_k1_IBM_s116
                                 + 0.00872163366990519 w_k1_IBM_s117
                                 + 0.0341146380855576 w_k1_INTC_s000
                                 + 0.0262436557069158 w_k1_INTC_s001
                                 + 0.0355612288188212 w_k1_INTC_s002
                                 + 0.0246185800445848 w_k1_INTC_s003
                                 + 0.0567823885835174 w_k1_INTC_s004
                                 + 0.024787799988618 w_k1_INTC_s005
                                 + 0.02888230075795 w_k1_INTC_s006
                                 + 0.0252950167068699 w_k1_INTC_s007
                                 + 0.0124479232996086 w_k1_INTC_s010
                                 + 0.0423731647800501 w_k1_INTC_s011
                                 + 0.0223149341410299 w_k1_INTC_s012
                                 + 0.00966194297897647 w_k1_INTC_s013
                                 + 0.0418039698497338 w_k1_INTC_s014
                                 + 0.0147247307952492 w_k1_INTC_s015
                                 + 0.0147398362731585 w_k1_INTC_s016
                                 + 0.0169993157222165 w_k1_INTC_s017
                                 + 0.102003221178637 w_k1_INTC_s100
                                 + 0.107245965142735 w_k1_INTC_s101
                                 + 0.0976838371485308 w_k1_INTC_s102
                                 + 0.030593624195696 w_k1_INTC_s103
                                 + 0.0386276203995417 w_k1_INTC_s104
                                 + 0.0381707777613157 w_k1_INTC_s105
                                 + 0.0285989417692076 w_k1_INTC_s106
                                 + 0.0356841468296234 w_k1_INTC_s107
                                 + 0.0129552010543858 w_k1_INTC_s110
                                 + 0.0120956392412331 w_k1_INTC_s111
                                 + 0.00921451223514695 w_k1_INTC_s112
                                 + 0.0103992407730873 w_k1_INTC_s113
                                 + 0.0246580339178185 w_k1_INTC_s114
                                 + 0.0235506647054258 w_k1_INTC_s115
                                 + 0.0268616190420896 w_k1_INTC_s116
                                 + 0.00873396669217973 w_k1_INTC_s117
                                 + 0.0317394469458807 w_k1_KO_s000
                                 + 0.0290678904450332 w_k1_KO_s001
                                 + 0.0396964655690126 w_k1_KO_s002
                                 + 0.0254707708536769 w_k1_KO_s003
                                 + 0.0495183779275485 w_k1_KO_s004
                                 + 0.0213359224927058 w_k1_KO_s005
                                 + 0.0288374972203884 w_k1_KO_s006
                                 + 0.0277793401295392 w_k1_KO_s007
                                 + 0.014167968144775 w_k1_KO_s010
                                 + 0.0401938977626455 w_k1_KO_s011
                                 + 0.0232296161641742 w_k1_KO_s012
                                 + 0.00897592656830225 w_k1_KO_s013
                                 + 0.0429359067618582 w_k1_KO_s014
                                 + 0.0149959720882827 w_k1_KO_s015
                                 + 0.0160446003958668 w_k1_KO_s016
                                 + 0.0145735400411046 w_k1_KO_s017
                                 + 0.0933893351661418 w_k1_KO_s100
                                 + 0.120398375637379 w_k1_KO_s101
                                 + 0.0868860366563607 w_k1_KO_s102
                                 + 0.0300089101151818 w_k1_KO_s103
                                 + 0.0316178224869969 w_k1_KO_s104
                                 + 0.0310289073274599 w_k1_KO_s105
                                 + 0.0316560204002735 w_k1_KO_s106
                                 + 0.0333991369721701 w_k1_KO_s107
                                 + 0.0145378469682979 w_k1_KO_s110
                                 + 0.0129562720529871 w_k1_KO_s111
                                 + 0.0100395766352366 w_k1_KO_s112
                                 + 0.0108538055201423 w_k1_KO_s113
                                 + 0.0279411749259657 w_k1_KO_s114
                                 + 0.0234158628911491 w_k1_KO_s115
                                 + 0.0264149497014237 w_k1_KO_s116
                                 + 0.00874252512803328 w_k1_KO_s117
                                 + 0.0294867364039484 w_k1_MAT_s000
                                 + 0.0265160589919087 w_k1_MAT_s001
                                 + 0.0473092690740965 w_k1_MAT_s002
                                 + 0.024364602071744 w_k1_MAT_s003
                                 + 0.0483252958658758 w_k1_MAT_s004
                                 + 0.0218043978325 w_k1_MAT_s005
                                 + 0.0355617046183862 w_k1_MAT_s006
                                 + 0.0334034201238123 w_k1_MAT_s007
                                 + 0.0167267940162711 w_k1_MAT_s010
                                 + 0.0381158490332887 w_k1_MAT_s011
                                 + 0.0237770080890886 w_k1_MAT_s012
                                 + 0.0101845867682226 w_k1_MAT_s013
                                 + 0.0413564621367386 w_k1_MAT_s014
                                 + 0.0142314753673782 w_k1_MAT_s015
                                 + 0.016912871163802 w_k1_MAT_s016
                                 + 0.0137539311090246 w_k1_MAT_s017
                                 + 0.101187370698912 w_k1_MAT_s100
                                 + 0.132174537932812 w_k1_MAT_s101
                                 + 0.0848847123853746 w_k1_MAT_s102
                                 + 0.0300894106175187 w_k1_MAT_s103
                                 + 0.028499627684556 w_k1_MAT_s104
                                 + 0.034063382807917 w_k1_MAT_s105
                                 + 0.0285408608358748 w_k1_MAT_s106
                                 + 0.0406865919034568 w_k1_MAT_s107
                                 + 0.0174291456165145 w_k1_MAT_s110
                                 + 0.0152969054972012 w_k1_MAT_s111
                                 + 0.0100590853020407 w_k1_MAT_s112
                                 + 0.00974825898034972 w_k1_MAT_s113
                                 + 0.0253157174568385 w_k1_MAT_s114
                                 + 0.0248894039876343 w_k1_MAT_s115
                                 + 0.0279465992380941 w_k1_MAT_s116
                                 + 0.00882141863944019 w_k1_MAT_s117
                                 + 0.0322974263832186 w_k1_MSFT_s000
                                 + 0.0264719629618464 w_k1_MSFT_s001
                                 + 0.0385318108566989 w_k1_MSFT_s002
                                 + 0.0265184329519904 w_k1_MSFT_s003
                                 + 0.0521819287654725 w_k1_MSFT_s004
                                 + 0.0236907818803714 w_k1_MSFT_s005
                                 + 0.0280395987732585 w_k1_MSFT_s006
                                 + 0.0278998516347331 w_k1_MSFT_s007
                                 + 0.012383770896418 w_k1_MSFT_s010
                                 + 0.0431162432120933 w_k1_MSFT_s011
                                 + 0.0226174026803074 w_k1_MSFT_s012
                                 + 0.00908961717898231 w_k1_MSFT_s013
                                 + 0.0416405316164798 w_k1_MSFT_s014
                                 + 0.0162878571153191 w_k1_MSFT_s015
                                 + 0.0147139340464927 w_k1_MSFT_s016
                                 + 0.0151840275488052 w_k1_MSFT_s017
                                 + 0.0913450072124582 w_k1_MSFT_s100
                                 + 0.103036175650438 w_k1_MSFT_s101
                                 + 0.0873152711358128 w_k1_MSFT_s102
                                 + 0.0320634314789772 w_k1_MSFT_s103
                                 + 0.0312421772773041 w_k1_MSFT_s104
                                 + 0.0315046718993644 w_k1_MSFT_s105
                                 + 0.0298783236821106 w_k1_MSFT_s106
                                 + 0.0334855301187019 w_k1_MSFT_s107
                                 + 0.0131885864051628 w_k1_MSFT_s110
                                 + 0.0121458869482704 w_k1_MSFT_s111
                                 + 0.00998907858077448 w_k1_MSFT_s112
                                 + 0.0105869036897014 w_k1_MSFT_s113
                                 + 0.0252779679344725 w_k1_MSFT_s114
                                 + 0.022790127389688 w_k1_MSFT_s115
                                 + 0.0252390385390689 w_k1_MSFT_s116
                                 + 0.00915205303502785 w_k1_MSFT_s117
                                 + 0.0281172378420693 w_k1_PG_s000
                                 + 0.0280712872915863 w_k1_PG_s001
                                 + 0.0369385008005832 w_k1_PG_s002
                                 + 0.0283103168267872 w_k1_PG_s003
                                 + 0.0514696417687107 w_k1_PG_s004
                                 + 0.0225598528080423 w_k1_PG_s005
                                 + 0.0302367080799733 w_k1_PG_s006
                                 + 0.0294255028159717 w_k1_PG_s007
                                 + 0.0119422703105011 w_k1_PG_s010
                                 + 0.0378549293298647 w_k1_PG_s011
                                 + 0.0241079775185867 w_k1_PG_s012
                                 + 0.00802572074123615 w_k1_PG_s013
                                 + 0.0392235572394208 w_k1_PG_s014
                                 + 0.0138047052122271 w_k1_PG_s015
                                 + 0.0156175834177366 w_k1_PG_s016
                                 + 0.0148691467182839 w_k1_PG_s017
                                 + 0.0995751615100545 w_k1_PG_s100
                                 + 0.110071673617328 w_k1_PG_s101
                                 + 0.0858052091895062 w_k1_PG_s102
                                 + 0.0320696060461037 w_k1_PG_s103
                                 + 0.0315660357460763 w_k1_PG_s104
                                 + 0.0297193219140981 w_k1_PG_s105
                                 + 0.0313620130799493 w_k1_PG_s106
                                 + 0.0344591771620599 w_k1_PG_s107
                                 + 0.014641520570075 w_k1_PG_s110
                                 + 0.0132549440646499 w_k1_PG_s111
                                 + 0.0110033051756168 w_k1_PG_s112
                                 + 0.0105980454996535 w_k1_PG_s113
                                 + 0.0216802570561387 w_k1_PG_s114
                                 + 0.0255889312518834 w_k1_PG_s115
                                 + 0.0252803960689848 w_k1_PG_s116
                                 + 0.00805918474395095 w_k1_PG_s117 >= 1.18
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
