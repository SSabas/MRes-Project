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
 w_k1_ADM_s0_return_constraint:  0.926853 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.15003 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.975694 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.1857 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.979872 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.872757 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.924532 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.21297 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.875625 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.13984 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.982339 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.57958 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.26117 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.03314 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.18023 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.01937 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.90497 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.42844 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.648191 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.770873 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.09734 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.852341 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.929027 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.67811 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.570221 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.00712 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.10572 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.872315 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.25502 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.810748 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.82142 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.37039 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.84668 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.976914 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.3086 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.33601 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.13015 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.729119 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.971557 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.24237 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.954906 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.0227 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.969975 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.09274 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.05824 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.44488 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.624808 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.32765 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.19136 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.80338 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08483 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.00592 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.09413 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.33486 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.927254 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.72817 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.517888 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.738002 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.57535 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.844251 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.12317 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.80944 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.537491 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.869475 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.992898 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.734948 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.46652 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.757407 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.589482 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.916876 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.74172 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.836352 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.50615 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.32067 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.12432 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.66286 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.938362 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.29437 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.00676 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.12294 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.985155 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.10225 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.746961 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.13072 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.637451 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.38623 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.03241 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.92484 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.19426 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.836175 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.5076 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.12535 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.941987 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 2.11238 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.464871 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.663646 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.57037 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.66845 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.42246 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 2.12116 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.427649 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.865953 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.991162 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.70886 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.33022 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.721472 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.487913 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.859269 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 2.08465 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.861402 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.90146 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.26729 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.27229 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.654868 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.980087 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.13302 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.934522 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.09714 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.04535 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.00553 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.899205 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26373 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.762325 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22055 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.914689 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.55467 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.15418 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.879399 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.23924 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.13392 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.96105 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.50769 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.62547 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.766041 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.36596 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.766819 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.13286 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.64488 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.529121 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.974568 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.91464 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.759772 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.25559 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.747307 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.653612 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.977901 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.51927 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.888126 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.49784 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.23519 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.07225 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.713847 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.907474 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.36711 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.25428 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.849718 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.851747 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.25802 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.701636 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.869728 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.743098 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.24583 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.56496 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.45188 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.906074 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.11828 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.980123 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.16533 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.984039 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.61109 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.572014 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.762829 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.47486 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.04098 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.01234 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.13032 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.755073 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.681516 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.33384 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.760709 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.13482 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.04719 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.880912 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.65315 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.69556 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.962057 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.15564 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.900062 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.38464 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.704662 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.951794 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.05115 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.963814 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.02947 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.20264 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.04128 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.09366 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.22455 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.847041 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.1876 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.06373 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.40863 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.02193 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.945882 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.28403 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05775 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.989348 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.4122 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.685076 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.831684 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.30046 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.890567 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.964738 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.26133 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.48486 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.794091 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.818307 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.790682 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.11218 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.745773 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.706973 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.01291 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.43921 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.892583 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.31321 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.31446 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.981245 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.77187 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.984296 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.32095 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.04639 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.874879 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.891096 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.08083 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.747754 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.03227 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.704894 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.36945 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.16449 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.42959 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.928704 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.00689 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.276 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.18846 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.11796 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.89657 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.611806 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.738005 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.40857 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.848921 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.16483 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.38413 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.661055 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.826157 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.14776 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.82504 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.15167 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.93291 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.703236 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.818755 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.76824 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.995894 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.27706 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.996832 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.20758 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.717996 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.939546 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.40696 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.20906 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.802779 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.711152 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.52332 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.765422 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.884721 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.499544 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.31301 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.37314 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.79124 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.857437 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.21468 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.873674 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.51024 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.984457 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.96126 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.515358 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.782672 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.67059 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.07323 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.44655 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.9104 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.966608 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.01183 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.79167 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.689378 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 2.06389 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.925759 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.665658 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.565151 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.51783 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.800302 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.14188 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.748236 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.1096 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.53167 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.954248 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.08443 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.00833 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.10678 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02356 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.04548 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.884536 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.1207 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.802283 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.13005 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.900373 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.45188 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.9949 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.854686 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.23538 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.980375 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.00744 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.48542 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.681636 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.839426 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.21493 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.713776 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.28284 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.72067 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.617425 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.958799 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.00876 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.84011 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.2907 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.770823 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.727363 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.979083 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.58299 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.92523 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.43187 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.25959 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.0473 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.820638 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.951081 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.19782 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.988239 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.0565 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.940504 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.00129 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.89781 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.07308 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.85006 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.22262 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.0503 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.34024 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05755 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.97181 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.18194 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.05606 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.932285 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.43612 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.671151 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.751038 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.19683 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.838496 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11156 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.28612 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.666542 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.864764 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.00853 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.793763 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.18878 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.883372 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.793202 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.927504 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.2922 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.980462 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.34034 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.00871 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04461 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.834432 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.146999957304537 z_k1_s0000
                                 - 0.43099973078284 z_k1_s0001
                                 - 0.152999845023205 z_k1_s0002
                                 - 0.271999013241998 z_k1_s0003
                                 - 0.403999005050122 z_k1_s0004
                                 - 0.393998371519867 z_k1_s0005
                                 - 0.159999796094899 z_k1_s0006
                                 - 0.500999241499777 z_k1_s0007
                                 - 0.19099977891096 z_k1_s0010
                                 - 0.208999845853302 z_k1_s0011
                                 - 0.281999431030985 z_k1_s0012
                                 - 0.401999141980368 z_k1_s0013
                                 - 0.235000628858501 z_k1_s0014
                                 - 0.175999870192254 z_k1_s0015
                                 - 0.552000435166116 z_k1_s0016
                                 - 0.159000003057704 z_k1_s0017
                                 - 0.945998713402838 z_k1_s0020
                                 - 0.701999121110465 z_k1_s0021
                                 - 2.72899989170979 z_k1_s0022
                                 - 2.13599810639665 z_k1_s0023
                                 - 0.600998866541215 z_k1_s0024
                                 - 0.630999609606472 z_k1_s0025
                                 - 1.32799389524545 z_k1_s0026
                                 - 1.80000109197125 z_k1_s0027
                                 - 0.499000686079807 z_k1_s0030
                                 - 0.230999551189911 z_k1_s0031
                                 - 0.280999449381244 z_k1_s0032
                                 - 0.340999183571155 z_k1_s0033
                                 - 0.57500044514682 z_k1_s0034
                                 - 0.6999992928771 z_k1_s0035
                                 - 0.263999333174518 z_k1_s0036
                                 - 0.6999992928771 z_k1_s0037
                                 - 0.404999446723136 z_k1_s0040
                                 - 0.522999667605163 z_k1_s0041
                                 - 0.202999973009198 z_k1_s0042
                                 - 0.189999885366307 z_k1_s0043
                                 - 0.524000166900423 z_k1_s0044
                                 - 0.19699991203557 z_k1_s0045
                                 - 0.378999805036979 z_k1_s0046
                                 - 0.246999797616237 z_k1_s0047
                                 - 0.12399990187296 z_k1_s0050
                                 - 0.585999713454381 z_k1_s0051
                                 - 0.524000085617739 z_k1_s0052
                                 - 0.456998872295083 z_k1_s0053
                                 - 0.372999376516406 z_k1_s0054
                                 - 0.285999898745634 z_k1_s0055
                                 - 0.492000277702052 z_k1_s0056
                                 - 0.388999280474249 z_k1_s0057
                                 - 0.271999803114075 z_k1_s0060
                                 - 0.597999807339582 z_k1_s0061
                                 - 0.616000281443078 z_k1_s0062
                                 - 0.247999883342196 z_k1_s0063
                                 - 0.265999746846164 z_k1_s0064
                                 - 0.48800030226004 z_k1_s0065
                                 - 0.312001094132773 z_k1_s0066
                                 - 0.25299977758224 z_k1_s0067
                                 - 0.474999453277798 z_k1_s0070
                                 - 1.12900128100005 z_k1_s0071
                                 - 0.283999917336971 z_k1_s0072
                                 - 0.715999963188785 z_k1_s0073
                                 - 0.852000281710303 z_k1_s0074
                                 - 0.336999519556986 z_k1_s0075
                                 - 0.729999918080796 z_k1_s0076
                                 - 0.773996749459169 z_k1_s0077
                                 - 0.198000158983298 z_k1_s0100
                                 - 0.263000140641885 z_k1_s0101
                                 - 0.203000076495415 z_k1_s0102
                                 - 0.588999505436719 z_k1_s0103
                                 - 0.34500089584273 z_k1_s0104
                                 - 0.298999019728605 z_k1_s0105
                                 - 0.305001555745788 z_k1_s0106
                                 - 0.432999444056003 z_k1_s0107
                                 - 0.749999608688882 z_k1_s0110
                                 - 0.352000345494575 z_k1_s0111
                                 - 1.55799874331321 z_k1_s0112
                                 - 0.838000655213601 z_k1_s0113
                                 - 0.976998525833638 z_k1_s0114
                                 - 0.815001836464451 z_k1_s0115
                                 - 0.585000846958618 z_k1_s0116
                                 - 0.526000843603055 z_k1_s0117
                                 - 0.403000620049582 z_k1_s0120
                                 - 0.513000911243433 z_k1_s0121
                                 - 1.1119997205152 z_k1_s0122
                                 - 0.529000458950946 z_k1_s0123
                                 - 1.60199958157674 z_k1_s0124
                                 - 1.62099964783024 z_k1_s0125
                                 - 1.26900257745526 z_k1_s0126
                                 - 1.72700509830419 z_k1_s0127
                                 - 0.424001412881081 z_k1_s0130
                                 - 0.484001110344929 z_k1_s0131
                                 - 0.16800004059958 z_k1_s0132
                                 - 0.212000114640004 z_k1_s0133
                                 - 0.364001715417234 z_k1_s0134
                                 - 0.618999689887915 z_k1_s0135
                                 - 0.335000530091832 z_k1_s0136
                                 - 0.353000143430718 z_k1_s0137
                                 - 0.221000061594106 z_k1_s0140
                                 - 0.308000724529617 z_k1_s0141
                                 - 0.453001358921933 z_k1_s0142
                                 - 0.297998831225903 z_k1_s0143
                                 - 0.218999965233485 z_k1_s0144
                                 - 0.407999896558256 z_k1_s0145
                                 - 0.348999171170837 z_k1_s0146
                                 - 0.566999793974314 z_k1_s0147
                                 - 0.98900037670109 z_k1_s0150
                                 - 0.428000523822416 z_k1_s0151
                                 - 1.65400362998799 z_k1_s0152
                                 - 0.369000284869185 z_k1_s0153
                                 - 1.12799977485771 z_k1_s0154
                                 - 0.648000854833991 z_k1_s0155
                                 - 0.696001407636925 z_k1_s0156
                                 - 0.696001407636925 z_k1_s0157
                                 - 0.918996593774494 z_k1_s0160
                                 - 0.269999438044787 z_k1_s0161
                                 - 0.660998420181341 z_k1_s0162
                                 - 0.598000680484797 z_k1_s0163
                                 - 0.565999584073928 z_k1_s0164
                                 - 0.269999438044787 z_k1_s0165
                                 - 1.32199684036268 z_k1_s0166
                                 - 0.269999438044787 z_k1_s0167
                                 - 1.25300457703073 z_k1_s0170
                                 - 1.12900455377623 z_k1_s0171
                                 - 0.675000746879526 z_k1_s0172
                                 - 0.470001842683805 z_k1_s0173
                                 - 0.259000722032367 z_k1_s0174
                                 - 0.787001831242228 z_k1_s0175
                                 - 0.447001678866675 z_k1_s0176
                                 - 0.47400148898107 z_k1_s0177
                                 - 0.456001265512313 z_k1_s1000
                                 - 0.325000184875035 z_k1_s1001
                                 - 0.25600034035724 z_k1_s1002
                                 - 0.359001713393435 z_k1_s1003
                                 - 0.224000297812585 z_k1_s1004
                                 - 0.140000120207768 z_k1_s1005
                                 - 0.597001865006561 z_k1_s1006
                                 - 0.279999713014752 z_k1_s1007
                                 - 0.172000091778164 z_k1_s1010
                                 - 0.0589999850459854 z_k1_s1011
                                 - 0.0509999754702554 z_k1_s1012
                                 - 0.0660000219497569 z_k1_s1013
                                 - 0.162999881330413 z_k1_s1014
                                 - 0.190000512673665 z_k1_s1015
                                 - 0.207000276297021 z_k1_s1016
                                 - 0.233000250368128 z_k1_s1017
                                 - 0.0790002289163739 z_k1_s1020
                                 - 0.0780000609554135 z_k1_s1021
                                 - 0.0440000226825414 z_k1_s1022
                                 - 0.12799998300376 z_k1_s1023
                                 - 0.0940001403308804 z_k1_s1024
                                 - 0.11599992347216 z_k1_s1025
                                 - 0.0320000283509388 z_k1_s1026
                                 - 0.0809999128383199 z_k1_s1027
                                 - 0.158999902291458 z_k1_s1030
                                 - 0.35200100683803 z_k1_s1031
                                 - 0.2460015706454 z_k1_s1032
                                 - 0.165000378455012 z_k1_s1033
                                 - 0.161000061012643 z_k1_s1034
                                 - 0.186001297024939 z_k1_s1035
                                 - 0.128000434123148 z_k1_s1036
                                 - 0.0990003766735101 z_k1_s1037
                                 - 0.283999595368436 z_k1_s1040
                                 - 0.156999651076491 z_k1_s1041
                                 - 0.184999673801409 z_k1_s1042
                                 - 0.137999656366679 z_k1_s1043
                                 - 0.35799853705106 z_k1_s1044
                                 - 0.176999584465902 z_k1_s1045
                                 - 0.482999570481661 z_k1_s1046
                                 - 0.150999632399754 z_k1_s1047
                                 - 0.241999660769645 z_k1_s1050
                                 - 0.315998359076266 z_k1_s1051
                                 - 0.442998442577286 z_k1_s1052
                                 - 0.498999443930574 z_k1_s1053
                                 - 0.497999061478401 z_k1_s1054
                                 - 0.147999693409688 z_k1_s1055
                                 - 0.170999813023097 z_k1_s1056
                                 - 0.234999535600484 z_k1_s1057
                                 - 0.146999820972068 z_k1_s1060
                                 - 0.20200020394761 z_k1_s1061
                                 - 0.14599986026338 z_k1_s1062
                                 - 0.389999940374503 z_k1_s1063
                                 - 0.325999741420925 z_k1_s1064
                                 - 0.116999812512502 z_k1_s1065
                                 - 0.0999999716652689 z_k1_s1066
                                 - 0.267999985118866 z_k1_s1067
                                 - 0.321999149906608 z_k1_s1070
                                 - 0.135000138770927 z_k1_s1071
                                 - 0.162000083045088 z_k1_s1072
                                 - 0.35399913023896 z_k1_s1073
                                 - 0.169000091786544 z_k1_s1074
                                 - 0.39699969077948 z_k1_s1075
                                 - 0.247000696034344 z_k1_s1076
                                 - 0.300999749782423 z_k1_s1077
                                 - 0.229000142723746 z_k1_s1100
                                 - 0.0909998434823848 z_k1_s1101
                                 - 0.134999711504758 z_k1_s1102
                                 - 0.270000063408137 z_k1_s1103
                                 - 0.184999954502063 z_k1_s1104
                                 - 0.134999711504758 z_k1_s1105
                                 - 0.254998725721669 z_k1_s1106
                                 - 0.300998558636031 z_k1_s1107
                                 - 0.0820000372813263 z_k1_s1110
                                 - 0.140999886073908 z_k1_s1111
                                 - 0.039000009545993 z_k1_s1112
                                 - 0.122999972021952 z_k1_s1113
                                 - 0.10299988107525 z_k1_s1114
                                 - 0.105000057969996 z_k1_s1115
                                 - 0.0939999240492722 z_k1_s1116
                                 - 0.152000019994632 z_k1_s1117
                                 - 0.0870002681647403 z_k1_s1120
                                 - 0.137000048213501 z_k1_s1121
                                 - 0.129999699326735 z_k1_s1122
                                 - 0.0410000094799537 z_k1_s1123
                                 - 0.0400000274389763 z_k1_s1124
                                 - 0.121999978598894 z_k1_s1125
                                 - 0.041999991520931 z_k1_s1126
                                 - 0.0789998694370076 z_k1_s1127
                                 - 0.710999876108068 z_k1_s1130
                                 - 0.216999802510755 z_k1_s1131
                                 - 0.175999908303596 z_k1_s1132
                                 - 0.4720007537588 z_k1_s1133
                                 - 0.150999762596969 z_k1_s1134
                                 - 0.428998922040736 z_k1_s1135
                                 - 0.41100050522654 z_k1_s1136
                                 - 0.405000042289311 z_k1_s1137
                                 - 0.0420000193234462 z_k1_s1140
                                 - 0.0980000890880361 z_k1_s1141
                                 - 0.118999966749774 z_k1_s1142
                                 - 0.0440000171007535 z_k1_s1143
                                 - 0.0449999829894108 z_k1_s1144
                                 - 0.112000007529195 z_k1_s1145
                                 - 0.118999966749774 z_k1_s1146
                                 - 0.0809998109809571 z_k1_s1147
                                 - 0.121999927870425 z_k1_s1150
                                 - 0.261000415159918 z_k1_s1151
                                 - 0.445999464021669 z_k1_s1152
                                 - 0.145999814133562 z_k1_s1153
                                 - 0.22899889987744 z_k1_s1154
                                 - 0.16699975926376 z_k1_s1155
                                 - 0.101999963117873 z_k1_s1156
                                 - 0.312999751997139 z_k1_s1157
                                 - 0.300001023711248 z_k1_s1160
                                 - 0.29900158243293 z_k1_s1161
                                 - 0.274000670455935 z_k1_s1162
                                 - 0.154000260971436 z_k1_s1163
                                 - 0.362001103011404 z_k1_s1164
                                 - 0.149000078576037 z_k1_s1165
                                 - 0.399999711612883 z_k1_s1166
                                 - 0.541999733235615 z_k1_s1167
                                 - 0.0769999445832672 z_k1_s1170
                                 - 0.0579999702146629 z_k1_s1171
                                 - 0.149999533331216 z_k1_s1172
                                 - 0.0969998823607627 z_k1_s1173
                                 - 0.0589999336235542 z_k1_s1174
                                 - 0.121000008573659 z_k1_s1175
                                 - 0.149000155822037 z_k1_s1176
                                 - 0.126000244117909 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.04259 w_k1_ADM_s000
                                 + 1.15157 w_k1_AXP_s000
                                 + 0.844997 w_k1_DIS_s000
                                 + 0.913272 w_k1_FDX_s000
                                 + 1.02662 w_k1_IBM_s000
                                 + 0.809413 w_k1_INTC_s000
                                 + 1.01972 w_k1_KO_s000 + 1.44375 w_k1_MAT_s000
                                 + 0.930845 w_k1_MSFT_s000
                                 + 1.00493 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.3855 w_k1_ADM_s000
                                 + 1.25004 w_k1_AXP_s000
                                 + 1.32969 w_k1_DIS_s000
                                 + 1.12241 w_k1_FDX_s000
                                 + 1.56927 w_k1_IBM_s000
                                 + 0.969241 w_k1_INTC_s000
                                 + 1.3681 w_k1_KO_s000 + 1.63866 w_k1_MAT_s000
                                 + 1.1802 w_k1_MSFT_s000 + 1.33046 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.26605 w_k1_ADM_s000
                                 + 1.17888 w_k1_AXP_s000
                                 + 1.04213 w_k1_DIS_s000
                                 + 1.11554 w_k1_FDX_s000
                                 + 0.809589 w_k1_IBM_s000
                                 + 0.723177 w_k1_INTC_s000
                                 + 0.936356 w_k1_KO_s000 + 1.1255 w_k1_MAT_s000
                                 + 0.923758 w_k1_MSFT_s000
                                 + 0.940819 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.28057 w_k1_ADM_s000
                                 + 1.23363 w_k1_AXP_s000
                                 + 1.50463 w_k1_DIS_s000
                                 + 1.43248 w_k1_FDX_s000
                                 + 0.988263 w_k1_IBM_s000
                                 + 0.88965 w_k1_INTC_s000 + 1.2508 w_k1_KO_s000
                                 + 1.291 w_k1_MAT_s000 + 1.23614 w_k1_MSFT_s000
                                 + 1.21528 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.818511 w_k1_ADM_s000
                                 + 1.07353 w_k1_AXP_s000
                                 + 1.17654 w_k1_DIS_s000
                                 + 1.09129 w_k1_FDX_s000
                                 + 1.18068 w_k1_IBM_s000
                                 + 0.966134 w_k1_INTC_s000
                                 + 1.11487 w_k1_KO_s000 + 1.58291 w_k1_MAT_s000
                                 + 0.991946 w_k1_MSFT_s000
                                 + 1.11871 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.804148 w_k1_ADM_s000
                                 + 0.72975 w_k1_AXP_s000
                                 + 0.619207 w_k1_DIS_s000
                                 + 0.837765 w_k1_FDX_s000
                                 + 0.746623 w_k1_IBM_s000
                                 + 0.694662 w_k1_INTC_s000
                                 + 0.659076 w_k1_KO_s000
                                 + 0.751336 w_k1_MAT_s000
                                 + 0.719849 w_k1_MSFT_s000
                                 + 0.843057 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.0157 w_k1_ADM_s000
                                 + 0.879305 w_k1_AXP_s000
                                 + 1.12432 w_k1_DIS_s000 + 1.1021 w_k1_FDX_s000
                                 + 0.783127 w_k1_IBM_s000
                                 + 0.832624 w_k1_INTC_s000
                                 + 0.95304 w_k1_KO_s000
                                 + 0.944169 w_k1_MAT_s000
                                 + 1.09462 w_k1_MSFT_s000
                                 + 1.05518 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.836908 w_k1_ADM_s000
                                 + 0.789647 w_k1_AXP_s000
                                 + 0.856152 w_k1_DIS_s000
                                 + 0.836024 w_k1_FDX_s000
                                 + 1.10795 w_k1_IBM_s000
                                 + 0.728108 w_k1_INTC_s000
                                 + 0.929476 w_k1_KO_s000
                                 + 0.895195 w_k1_MAT_s000
                                 + 0.806459 w_k1_MSFT_s000
                                 + 0.899089 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.901734 w_k1_ADM_s001
                                 + 0.768757 w_k1_AXP_s001
                                 + 0.881362 w_k1_DIS_s001
                                 + 0.995232 w_k1_FDX_s001
                                 + 0.668522 w_k1_IBM_s001
                                 + 1.03512 w_k1_INTC_s001
                                 + 0.752823 w_k1_KO_s001
                                 + 0.66799 w_k1_MAT_s001
                                 + 0.971038 w_k1_MSFT_s001
                                 + 1.01292 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.3022 w_k1_ADM_s001
                                 + 1.62146 w_k1_AXP_s001
                                 + 1.69558 w_k1_DIS_s001
                                 + 1.37383 w_k1_FDX_s001
                                 + 1.50475 w_k1_IBM_s001
                                 + 1.22403 w_k1_INTC_s001
                                 + 1.53574 w_k1_KO_s001 + 1.93151 w_k1_MAT_s001
                                 + 1.3163 w_k1_MSFT_s001 + 1.51761 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.616228 w_k1_ADM_s001
                                 + 0.74606 w_k1_AXP_s001
                                 + 0.618148 w_k1_DIS_s001
                                 + 0.729417 w_k1_FDX_s001
                                 + 0.777139 w_k1_IBM_s001
                                 + 0.725913 w_k1_INTC_s001
                                 + 0.630555 w_k1_KO_s001
                                 + 0.943435 w_k1_MAT_s001
                                 + 0.77359 w_k1_MSFT_s001
                                 + 0.859543 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.37832 w_k1_ADM_s001
                                 + 1.28062 w_k1_AXP_s001 + 1.3201 w_k1_DIS_s001
                                 + 1.13973 w_k1_FDX_s001
                                 + 1.17158 w_k1_IBM_s001
                                 + 1.08706 w_k1_INTC_s001
                                 + 1.13421 w_k1_KO_s001 + 1.38329 w_k1_MAT_s001
                                 + 1.1074 w_k1_MSFT_s001 + 1.16625 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.869105 w_k1_ADM_s001
                                 + 1.00528 w_k1_AXP_s001
                                 + 0.963434 w_k1_DIS_s001
                                 + 0.826997 w_k1_FDX_s001
                                 + 1.42952 w_k1_IBM_s001
                                 + 0.968382 w_k1_INTC_s001
                                 + 1.0533 w_k1_KO_s001 + 1.59929 w_k1_MAT_s001
                                 + 0.921626 w_k1_MSFT_s001
                                 + 0.998616 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0181 w_k1_ADM_s001
                                 + 0.877196 w_k1_AXP_s001
                                 + 1.3495 w_k1_DIS_s001 + 1.19957 w_k1_FDX_s001
                                 + 0.819713 w_k1_IBM_s001
                                 + 1.01425 w_k1_INTC_s001
                                 + 0.895492 w_k1_KO_s001
                                 + 0.854923 w_k1_MAT_s001
                                 + 1.21518 w_k1_MSFT_s001
                                 + 1.11961 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.836486 w_k1_ADM_s001
                                 + 1.04098 w_k1_AXP_s001
                                 + 0.893272 w_k1_DIS_s001
                                 + 0.99539 w_k1_FDX_s001
                                 + 0.95535 w_k1_IBM_s001
                                 + 1.0105 w_k1_INTC_s001
                                 + 0.789022 w_k1_KO_s001
                                 + 0.900676 w_k1_MAT_s001
                                 + 0.907071 w_k1_MSFT_s001
                                 + 0.890955 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.871515 w_k1_ADM_s001
                                 + 0.994978 w_k1_AXP_s001
                                 + 1.21941 w_k1_DIS_s001
                                 + 0.912173 w_k1_FDX_s001
                                 + 1.15203 w_k1_IBM_s001
                                 + 0.934385 w_k1_INTC_s001
                                 + 1.05435 w_k1_KO_s001 + 1.26238 w_k1_MAT_s001
                                 + 0.993618 w_k1_MSFT_s001
                                 + 1.06257 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.643093 w_k1_ADM_s002
                                 + 0.613383 w_k1_AXP_s002
                                 + 0.495181 w_k1_DIS_s002
                                 + 0.609624 w_k1_FDX_s002
                                 + 0.745382 w_k1_IBM_s002
                                 + 0.619962 w_k1_INTC_s002
                                 + 0.620657 w_k1_KO_s002
                                 + 0.845876 w_k1_MAT_s002
                                 + 0.670254 w_k1_MSFT_s002
                                 + 0.826269 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.839697 w_k1_ADM_s002
                                 + 1.40884 w_k1_AXP_s002
                                 + 0.910083 w_k1_DIS_s002
                                 + 0.921909 w_k1_FDX_s002
                                 + 1.49773 w_k1_IBM_s002
                                 + 0.892732 w_k1_INTC_s002
                                 + 1.20205 w_k1_KO_s002 + 2.33981 w_k1_MAT_s002
                                 + 0.881639 w_k1_MSFT_s002
                                 + 1.06848 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.951769 w_k1_ADM_s002
                                 + 1.15397 w_k1_AXP_s002
                                 + 1.02813 w_k1_DIS_s002
                                 + 1.02962 w_k1_FDX_s002
                                 + 1.01305 w_k1_IBM_s002
                                 + 0.886682 w_k1_INTC_s002
                                 + 1.22371 w_k1_KO_s002 + 1.45369 w_k1_MAT_s002
                                 + 1.00045 w_k1_MSFT_s002 + 1.0456 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.0347 w_k1_ADM_s002
                                 + 1.17682 w_k1_AXP_s002
                                 + 1.36917 w_k1_DIS_s002
                                 + 1.15712 w_k1_FDX_s002
                                 + 1.30103 w_k1_IBM_s002
                                 + 1.17554 w_k1_INTC_s002
                                 + 1.37792 w_k1_KO_s002 + 1.24172 w_k1_MAT_s002
                                 + 1.11618 w_k1_MSFT_s002
                                 + 1.17863 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.680986 w_k1_ADM_s002
                                 + 0.876811 w_k1_AXP_s002
                                 + 0.648713 w_k1_DIS_s002
                                 + 0.815773 w_k1_FDX_s002
                                 + 0.638602 w_k1_IBM_s002
                                 + 1.01329 w_k1_INTC_s002
                                 + 0.783204 w_k1_KO_s002
                                 + 0.89797 w_k1_MAT_s002
                                 + 0.889941 w_k1_MSFT_s002
                                 + 0.861638 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.37648 w_k1_ADM_s002
                                 + 1.70484 w_k1_AXP_s002
                                 + 1.81504 w_k1_DIS_s002
                                 + 1.47564 w_k1_FDX_s002
                                 + 1.45115 w_k1_IBM_s002
                                 + 1.53547 w_k1_INTC_s002
                                 + 1.53345 w_k1_KO_s002 + 2.23859 w_k1_MAT_s002
                                 + 1.48409 w_k1_MSFT_s002 + 1.3225 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.794148 w_k1_ADM_s002
                                 + 0.874625 w_k1_AXP_s002
                                 + 0.718535 w_k1_DIS_s002
                                 + 0.82337 w_k1_FDX_s002
                                 + 0.87866 w_k1_IBM_s002
                                 + 0.854935 w_k1_INTC_s002
                                 + 0.838577 w_k1_KO_s002
                                 + 1.03037 w_k1_MAT_s002
                                 + 0.776632 w_k1_MSFT_s002
                                 + 0.86052 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.933132 w_k1_ADM_s002
                                 + 1.02885 w_k1_AXP_s002
                                 + 1.02857 w_k1_DIS_s002
                                 + 1.08885 w_k1_FDX_s002
                                 + 0.620405 w_k1_IBM_s002
                                 + 1.04607 w_k1_INTC_s002
                                 + 0.836903 w_k1_KO_s002
                                 + 1.04589 w_k1_MAT_s002
                                 + 1.0618 w_k1_MSFT_s002
                                 + 0.972204 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.01118 w_k1_ADM_s003
                                 + 0.955997 w_k1_AXP_s003
                                 + 1.19023 w_k1_DIS_s003
                                 + 1.15706 w_k1_FDX_s003
                                 + 1.00323 w_k1_IBM_s003
                                 + 1.20401 w_k1_INTC_s003
                                 + 0.924449 w_k1_KO_s003
                                 + 0.810689 w_k1_MAT_s003
                                 + 1.15765 w_k1_MSFT_s003
                                 + 1.01301 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.967767 w_k1_ADM_s003
                                 + 0.865517 w_k1_AXP_s003
                                 + 0.874988 w_k1_DIS_s003
                                 + 1.02364 w_k1_FDX_s003
                                 + 0.750795 w_k1_IBM_s003
                                 + 1.15215 w_k1_INTC_s003
                                 + 0.802295 w_k1_KO_s003
                                 + 0.658284 w_k1_MAT_s003
                                 + 1.09552 w_k1_MSFT_s003
                                 + 0.924454 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.17232 w_k1_ADM_s003
                                 + 1.24272 w_k1_AXP_s003
                                 + 0.983914 w_k1_DIS_s003
                                 + 1.03132 w_k1_FDX_s003
                                 + 1.01296 w_k1_IBM_s003
                                 + 0.997719 w_k1_INTC_s003
                                 + 0.998052 w_k1_KO_s003
                                 + 1.28109 w_k1_MAT_s003
                                 + 1.00578 w_k1_MSFT_s003
                                 + 1.13035 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.06323 w_k1_ADM_s003
                                 + 0.859035 w_k1_AXP_s003
                                 + 0.907715 w_k1_DIS_s003
                                 + 0.916141 w_k1_FDX_s003
                                 + 0.861328 w_k1_IBM_s003
                                 + 0.950118 w_k1_INTC_s003
                                 + 0.820637 w_k1_KO_s003
                                 + 0.74485 w_k1_MAT_s003
                                 + 0.962901 w_k1_MSFT_s003
                                 + 0.958581 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.9089 w_k1_ADM_s003
                                 + 0.753793 w_k1_AXP_s003
                                 + 0.764819 w_k1_DIS_s003
                                 + 0.763276 w_k1_FDX_s003
                                 + 0.818199 w_k1_IBM_s003
                                 + 1.0214 w_k1_INTC_s003
                                 + 0.729651 w_k1_KO_s003
                                 + 0.640058 w_k1_MAT_s003
                                 + 0.822555 w_k1_MSFT_s003
                                 + 0.81068 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.17239 w_k1_ADM_s003
                                 + 1.1568 w_k1_AXP_s003 + 1.37923 w_k1_DIS_s003
                                 + 1.09151 w_k1_FDX_s003
                                 + 1.55861 w_k1_IBM_s003 + 1.052 w_k1_INTC_s003
                                 + 1.28927 w_k1_KO_s003 + 1.5834 w_k1_MAT_s003
                                 + 1.12915 w_k1_MSFT_s003
                                 + 1.20615 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.43701 w_k1_ADM_s003
                                 + 1.36307 w_k1_AXP_s003
                                 + 1.44068 w_k1_DIS_s003
                                 + 1.32627 w_k1_FDX_s003
                                 + 1.15527 w_k1_IBM_s003
                                 + 1.38944 w_k1_INTC_s003
                                 + 1.08381 w_k1_KO_s003
                                 + 0.954985 w_k1_MAT_s003
                                 + 1.23596 w_k1_MSFT_s003
                                 + 1.09017 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.924755 w_k1_ADM_s003
                                 + 0.836547 w_k1_AXP_s003
                                 + 0.817979 w_k1_DIS_s003
                                 + 0.852758 w_k1_FDX_s003
                                 + 1.09254 w_k1_IBM_s003
                                 + 0.887569 w_k1_INTC_s003
                                 + 0.877115 w_k1_KO_s003
                                 + 1.18496 w_k1_MAT_s003
                                 + 0.899673 w_k1_MSFT_s003
                                 + 0.9162 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.27589 w_k1_ADM_s004
                                 + 1.28068 w_k1_AXP_s004
                                 + 1.20612 w_k1_DIS_s004
                                 + 1.10431 w_k1_FDX_s004
                                 + 1.57967 w_k1_IBM_s004
                                 + 1.09254 w_k1_INTC_s004
                                 + 1.24819 w_k1_KO_s004 + 1.53946 w_k1_MAT_s004
                                 + 1.16681 w_k1_MSFT_s004
                                 + 1.19671 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.792454 w_k1_ADM_s004
                                 + 0.746479 w_k1_AXP_s004
                                 + 0.76726 w_k1_DIS_s004
                                 + 0.774341 w_k1_FDX_s004
                                 + 0.875236 w_k1_IBM_s004
                                 + 0.735505 w_k1_INTC_s004
                                 + 0.844073 w_k1_KO_s004
                                 + 0.741026 w_k1_MAT_s004
                                 + 0.750144 w_k1_MSFT_s004
                                 + 0.827427 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.20752 w_k1_ADM_s004
                                 + 1.58349 w_k1_AXP_s004
                                 + 1.36227 w_k1_DIS_s004
                                 + 1.37913 w_k1_FDX_s004
                                 + 0.947061 w_k1_IBM_s004
                                 + 1.23704 w_k1_INTC_s004
                                 + 1.15641 w_k1_KO_s004 + 1.39192 w_k1_MAT_s004
                                 + 1.20986 w_k1_MSFT_s004
                                 + 1.10014 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.817548 w_k1_ADM_s004
                                 + 0.643783 w_k1_AXP_s004
                                 + 0.773197 w_k1_DIS_s004
                                 + 0.85476 w_k1_FDX_s004
                                 + 0.595491 w_k1_IBM_s004
                                 + 0.729583 w_k1_INTC_s004
                                 + 0.774832 w_k1_KO_s004
                                 + 0.53209 w_k1_MAT_s004
                                 + 0.892718 w_k1_MSFT_s004
                                 + 0.846684 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09268 w_k1_ADM_s004
                                 + 0.879898 w_k1_AXP_s004
                                 + 0.9065 w_k1_DIS_s004
                                 + 0.932352 w_k1_FDX_s004
                                 + 1.10544 w_k1_IBM_s004
                                 + 0.941017 w_k1_INTC_s004
                                 + 1.07008 w_k1_KO_s004 + 1.30871 w_k1_MAT_s004
                                 + 1.07927 w_k1_MSFT_s004
                                 + 0.949794 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.21016 w_k1_ADM_s004
                                 + 1.10589 w_k1_AXP_s004
                                 + 1.14199 w_k1_DIS_s004
                                 + 1.17138 w_k1_FDX_s004
                                 + 0.979848 w_k1_IBM_s004
                                 + 1.25494 w_k1_INTC_s004
                                 + 1.10187 w_k1_KO_s004 + 0.88248 w_k1_MAT_s004
                                 + 1.23237 w_k1_MSFT_s004
                                 + 1.13603 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.10756 w_k1_ADM_s004
                                 + 0.946885 w_k1_AXP_s004
                                 + 0.944029 w_k1_DIS_s004
                                 + 1.1403 w_k1_FDX_s004
                                 + 0.743628 w_k1_IBM_s004
                                 + 0.996461 w_k1_INTC_s004
                                 + 0.868324 w_k1_KO_s004
                                 + 0.871098 w_k1_MAT_s004
                                 + 0.95338 w_k1_MSFT_s004
                                 + 0.986345 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.33023 w_k1_ADM_s004
                                 + 1.3286 w_k1_AXP_s004 + 1.59897 w_k1_DIS_s004
                                 + 1.29444 w_k1_FDX_s004
                                 + 1.22804 w_k1_IBM_s004
                                 + 0.939689 w_k1_INTC_s004
                                 + 1.23297 w_k1_KO_s004 + 1.71519 w_k1_MAT_s004
                                 + 1.24506 w_k1_MSFT_s004
                                 + 1.31634 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.05018 w_k1_ADM_s005
                                 + 1.58232 w_k1_AXP_s005
                                 + 1.47465 w_k1_DIS_s005
                                 + 1.16979 w_k1_FDX_s005
                                 + 2.23523 w_k1_IBM_s005
                                 + 1.30943 w_k1_INTC_s005
                                 + 1.72096 w_k1_KO_s005 + 2.15273 w_k1_MAT_s005
                                 + 1.23598 w_k1_MSFT_s005
                                 + 1.34986 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.10972 w_k1_ADM_s005
                                 + 1.35654 w_k1_AXP_s005
                                 + 1.45774 w_k1_DIS_s005
                                 + 1.21081 w_k1_FDX_s005
                                 + 1.39579 w_k1_IBM_s005
                                 + 1.25083 w_k1_INTC_s005
                                 + 1.39287 w_k1_KO_s005 + 1.24113 w_k1_MAT_s005
                                 + 1.22684 w_k1_MSFT_s005
                                 + 1.22642 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.768524 w_k1_ADM_s005
                                 + 1.18804 w_k1_AXP_s005
                                 + 0.911621 w_k1_DIS_s005
                                 + 0.933459 w_k1_FDX_s005
                                 + 0.937557 w_k1_IBM_s005
                                 + 0.748116 w_k1_INTC_s005
                                 + 1.00381 w_k1_KO_s005 + 1.1209 w_k1_MAT_s005
                                 + 0.832659 w_k1_MSFT_s005
                                 + 0.959338 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.86693 w_k1_ADM_s005
                                 + 1.07516 w_k1_AXP_s005
                                 + 1.35281 w_k1_DIS_s005
                                 + 1.13718 w_k1_FDX_s005
                                 + 0.975634 w_k1_IBM_s005
                                 + 0.9417 w_k1_INTC_s005 + 1.18536 w_k1_KO_s005
                                 + 1.08597 w_k1_MAT_s005
                                 + 1.09477 w_k1_MSFT_s005 + 1.1401 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.967005 w_k1_ADM_s005
                                 + 0.876038 w_k1_AXP_s005
                                 + 0.697922 w_k1_DIS_s005
                                 + 0.871692 w_k1_FDX_s005
                                 + 0.692264 w_k1_IBM_s005
                                 + 0.867865 w_k1_INTC_s005
                                 + 0.845263 w_k1_KO_s005
                                 + 0.608788 w_k1_MAT_s005
                                 + 0.821342 w_k1_MSFT_s005
                                 + 0.986544 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.22944 w_k1_ADM_s005
                                 + 1.15818 w_k1_AXP_s005
                                 + 1.13859 w_k1_DIS_s005 + 1.3062 w_k1_FDX_s005
                                 + 0.729014 w_k1_IBM_s005
                                 + 1.02675 w_k1_INTC_s005
                                 + 0.917072 w_k1_KO_s005
                                 + 0.729073 w_k1_MAT_s005
                                 + 0.968819 w_k1_MSFT_s005
                                 + 1.13356 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.73226 w_k1_ADM_s005
                                 + 0.876779 w_k1_AXP_s005
                                 + 0.811109 w_k1_DIS_s005
                                 + 0.803609 w_k1_FDX_s005
                                 + 1.25511 w_k1_IBM_s005
                                 + 1.02052 w_k1_INTC_s005
                                 + 1.11687 w_k1_KO_s005 + 1.17541 w_k1_MAT_s005
                                 + 0.871703 w_k1_MSFT_s005
                                 + 0.996349 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.799114 w_k1_ADM_s005
                                 + 0.73072 w_k1_AXP_s005
                                 + 0.809272 w_k1_DIS_s005
                                 + 0.833017 w_k1_FDX_s005
                                 + 0.692472 w_k1_IBM_s005
                                 + 0.897501 w_k1_INTC_s005
                                 + 0.913128 w_k1_KO_s005
                                 + 0.619872 w_k1_MAT_s005
                                 + 0.955422 w_k1_MSFT_s005
                                 + 0.984574 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.641154 w_k1_ADM_s006
                                 + 0.761045 w_k1_AXP_s006
                                 + 0.899577 w_k1_DIS_s006
                                 + 0.907286 w_k1_FDX_s006
                                 + 0.869119 w_k1_IBM_s006
                                 + 0.940884 w_k1_INTC_s006
                                 + 0.95778 w_k1_KO_s006
                                 + 0.694973 w_k1_MAT_s006
                                 + 0.999243 w_k1_MSFT_s006
                                 + 0.925731 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.16686 w_k1_ADM_s006
                                 + 1.0291 w_k1_AXP_s006 + 1.05154 w_k1_DIS_s006
                                 + 1.02129 w_k1_FDX_s006
                                 + 0.896805 w_k1_IBM_s006
                                 + 1.059 w_k1_INTC_s006 + 1.22676 w_k1_KO_s006
                                 + 0.922326 w_k1_MAT_s006
                                 + 1.12553 w_k1_MSFT_s006
                                 + 1.06161 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.979353 w_k1_ADM_s006
                                 + 0.989913 w_k1_AXP_s006
                                 + 1.04127 w_k1_DIS_s006
                                 + 0.884542 w_k1_FDX_s006
                                 + 1.31862 w_k1_IBM_s006
                                 + 0.899656 w_k1_INTC_s006
                                 + 1.14859 w_k1_KO_s006 + 1.02695 w_k1_MAT_s006
                                 + 0.956311 w_k1_MSFT_s006
                                 + 1.01709 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02818 w_k1_ADM_s006
                                 + 0.770174 w_k1_AXP_s006
                                 + 0.919231 w_k1_DIS_s006
                                 + 0.999193 w_k1_FDX_s006
                                 + 0.639225 w_k1_IBM_s006
                                 + 0.920824 w_k1_INTC_s006
                                 + 0.86565 w_k1_KO_s006
                                 + 0.500367 w_k1_MAT_s006
                                 + 1.03135 w_k1_MSFT_s006
                                 + 0.908153 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.32103 w_k1_ADM_s006
                                 + 1.46834 w_k1_AXP_s006
                                 + 1.77063 w_k1_DIS_s006
                                 + 1.29335 w_k1_FDX_s006
                                 + 1.50557 w_k1_IBM_s006
                                 + 1.57475 w_k1_INTC_s006
                                 + 1.84559 w_k1_KO_s006 + 1.68026 w_k1_MAT_s006
                                 + 1.48192 w_k1_MSFT_s006
                                 + 1.29714 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.41239 w_k1_ADM_s006
                                 + 1.43499 w_k1_AXP_s006
                                 + 1.36693 w_k1_DIS_s006 + 1.2938 w_k1_FDX_s006
                                 + 1.00366 w_k1_IBM_s006
                                 + 1.3271 w_k1_INTC_s006 + 1.25837 w_k1_KO_s006
                                 + 1.25444 w_k1_MAT_s006
                                 + 1.43128 w_k1_MSFT_s006
                                 + 1.15556 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.71985 w_k1_ADM_s006
                                 + 0.705127 w_k1_AXP_s006
                                 + 0.63587 w_k1_DIS_s006
                                 + 0.72431 w_k1_FDX_s006
                                 + 0.779305 w_k1_IBM_s006
                                 + 0.958812 w_k1_INTC_s006
                                 + 0.838754 w_k1_KO_s006
                                 + 0.517203 w_k1_MAT_s006
                                 + 0.806577 w_k1_MSFT_s006
                                 + 0.912898 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.816894 w_k1_ADM_s006
                                 + 0.8567 w_k1_AXP_s006
                                 + 0.754239 w_k1_DIS_s006
                                 + 0.846016 w_k1_FDX_s006
                                 + 0.891012 w_k1_IBM_s006
                                 + 0.840377 w_k1_INTC_s006
                                 + 0.872007 w_k1_KO_s006
                                 + 0.854405 w_k1_MAT_s006
                                 + 1.05742 w_k1_MSFT_s006
                                 + 0.977144 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.844995 w_k1_ADM_s007
                                 + 0.704691 w_k1_AXP_s007
                                 + 0.69818 w_k1_DIS_s007
                                 + 0.743363 w_k1_FDX_s007 + 1.077 w_k1_IBM_s007
                                 + 0.884309 w_k1_INTC_s007
                                 + 0.88033 w_k1_KO_s007
                                 + 0.713168 w_k1_MAT_s007
                                 + 0.811499 w_k1_MSFT_s007
                                 + 0.842976 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.92367 w_k1_ADM_s007
                                 + 0.935456 w_k1_AXP_s007
                                 + 1.06926 w_k1_DIS_s007
                                 + 1.01365 w_k1_FDX_s007
                                 + 0.820348 w_k1_IBM_s007
                                 + 1.03867 w_k1_INTC_s007
                                 + 0.846163 w_k1_KO_s007
                                 + 0.633973 w_k1_MAT_s007
                                 + 0.944896 w_k1_MSFT_s007
                                 + 1.03203 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.05307 w_k1_ADM_s007
                                 + 0.903621 w_k1_AXP_s007
                                 + 0.972122 w_k1_DIS_s007
                                 + 1.04046 w_k1_FDX_s007 + 1.0746 w_k1_IBM_s007
                                 + 1.20859 w_k1_INTC_s007
                                 + 0.953238 w_k1_KO_s007 + 0.7277 w_k1_MAT_s007
                                 + 1.11677 w_k1_MSFT_s007
                                 + 0.969763 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.17889 w_k1_ADM_s007
                                 + 1.36577 w_k1_AXP_s007
                                 + 1.49818 w_k1_DIS_s007
                                 + 1.34173 w_k1_FDX_s007
                                 + 1.03966 w_k1_IBM_s007
                                 + 1.17351 w_k1_INTC_s007
                                 + 1.11064 w_k1_KO_s007
                                 + 0.808324 w_k1_MAT_s007
                                 + 1.28993 w_k1_MSFT_s007
                                 + 1.30151 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.759163 w_k1_ADM_s007
                                 + 0.67149 w_k1_AXP_s007
                                 + 0.694895 w_k1_DIS_s007
                                 + 0.814402 w_k1_FDX_s007
                                 + 0.699609 w_k1_IBM_s007
                                 + 0.811631 w_k1_INTC_s007
                                 + 0.780589 w_k1_KO_s007
                                 + 0.596669 w_k1_MAT_s007
                                 + 0.794145 w_k1_MSFT_s007
                                 + 0.854453 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.40748 w_k1_ADM_s007
                                 + 1.19337 w_k1_AXP_s007
                                 + 1.50394 w_k1_DIS_s007 + 1.2137 w_k1_FDX_s007
                                 + 1.62842 w_k1_IBM_s007
                                 + 1.04412 w_k1_INTC_s007
                                 + 1.44995 w_k1_KO_s007 + 1.52565 w_k1_MAT_s007
                                 + 1.30705 w_k1_MSFT_s007
                                 + 1.36028 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.15357 w_k1_ADM_s007
                                 + 1.03232 w_k1_AXP_s007
                                 + 1.23783 w_k1_DIS_s007
                                 + 1.09144 w_k1_FDX_s007
                                 + 1.28189 w_k1_IBM_s007
                                 + 1.09444 w_k1_INTC_s007
                                 + 1.20659 w_k1_KO_s007 + 1.41237 w_k1_MAT_s007
                                 + 1.16576 w_k1_MSFT_s007
                                 + 1.09481 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.725401 w_k1_ADM_s007
                                 + 0.816398 w_k1_AXP_s007
                                 + 1.11097 w_k1_DIS_s007
                                 + 0.944314 w_k1_FDX_s007
                                 + 1.15982 w_k1_IBM_s007
                                 + 1.02048 w_k1_INTC_s007
                                 + 0.965051 w_k1_KO_s007
                                 + 0.872371 w_k1_MAT_s007
                                 + 0.986233 w_k1_MSFT_s007
                                 + 1.02915 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.781223 w_k1_ADM_s010
                                 + 1.00465 w_k1_AXP_s010
                                 + 1.04137 w_k1_DIS_s010
                                 + 0.916908 w_k1_FDX_s010
                                 + 1.18948 w_k1_IBM_s010
                                 + 0.985075 w_k1_INTC_s010
                                 + 1.12562 w_k1_KO_s010 + 1.15305 w_k1_MAT_s010
                                 + 1.04607 w_k1_MSFT_s010
                                 + 1.03102 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.04993 w_k1_ADM_s010
                                 + 1.23618 w_k1_AXP_s010
                                 + 0.974822 w_k1_DIS_s010
                                 + 1.04692 w_k1_FDX_s010
                                 + 1.33806 w_k1_IBM_s010
                                 + 0.887835 w_k1_INTC_s010
                                 + 1.02356 w_k1_KO_s010 + 1.63516 w_k1_MAT_s010
                                 + 0.968634 w_k1_MSFT_s010
                                 + 0.896887 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.692092 w_k1_ADM_s010
                                 + 0.763889 w_k1_AXP_s010
                                 + 0.669444 w_k1_DIS_s010
                                 + 0.674425 w_k1_FDX_s010
                                 + 1.0693 w_k1_IBM_s010
                                 + 0.743536 w_k1_INTC_s010
                                 + 0.989002 w_k1_KO_s010
                                 + 1.20164 w_k1_MAT_s010
                                 + 0.894354 w_k1_MSFT_s010
                                 + 0.812286 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10938 w_k1_ADM_s010
                                 + 1.27311 w_k1_AXP_s010 + 1.1547 w_k1_DIS_s010
                                 + 1.27932 w_k1_FDX_s010 + 0.8995 w_k1_IBM_s010
                                 + 0.996619 w_k1_INTC_s010
                                 + 1.12601 w_k1_KO_s010 + 1.38229 w_k1_MAT_s010
                                 + 1.22068 w_k1_MSFT_s010
                                 + 1.07994 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.13588 w_k1_ADM_s010
                                 + 0.981964 w_k1_AXP_s010
                                 + 0.901711 w_k1_DIS_s010
                                 + 1.00049 w_k1_FDX_s010
                                 + 0.741238 w_k1_IBM_s010
                                 + 0.942474 w_k1_INTC_s010
                                 + 0.801592 w_k1_KO_s010
                                 + 0.671647 w_k1_MAT_s010
                                 + 0.990509 w_k1_MSFT_s010
                                 + 0.820159 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.900746 w_k1_ADM_s010
                                 + 0.714928 w_k1_AXP_s010
                                 + 0.640196 w_k1_DIS_s010
                                 + 0.828465 w_k1_FDX_s010
                                 + 0.735997 w_k1_IBM_s010
                                 + 0.836742 w_k1_INTC_s010
                                 + 0.757814 w_k1_KO_s010
                                 + 0.777095 w_k1_MAT_s010
                                 + 0.774793 w_k1_MSFT_s010
                                 + 0.748673 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.12189 w_k1_ADM_s010
                                 + 1.26307 w_k1_AXP_s010
                                 + 1.57957 w_k1_DIS_s010 + 1.2266 w_k1_FDX_s010
                                 + 1.45776 w_k1_IBM_s010
                                 + 1.20293 w_k1_INTC_s010
                                 + 1.44308 w_k1_KO_s010 + 2.40257 w_k1_MAT_s010
                                 + 1.28777 w_k1_MSFT_s010
                                 + 1.28453 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.739501 w_k1_ADM_s010
                                 + 0.904472 w_k1_AXP_s010
                                 + 0.87398 w_k1_DIS_s010
                                 + 0.947521 w_k1_FDX_s010
                                 + 0.934334 w_k1_IBM_s010
                                 + 0.912545 w_k1_INTC_s010
                                 + 0.927303 w_k1_KO_s010
                                 + 1.50128 w_k1_MAT_s010
                                 + 1.03039 w_k1_MSFT_s010
                                 + 0.856034 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.3196 w_k1_ADM_s011
                                 + 1.23054 w_k1_AXP_s011
                                 + 1.12062 w_k1_DIS_s011 + 1.2237 w_k1_FDX_s011
                                 + 0.853479 w_k1_IBM_s011
                                 + 1.01506 w_k1_INTC_s011
                                 + 0.966132 w_k1_KO_s011
                                 + 0.991735 w_k1_MAT_s011
                                 + 1.30543 w_k1_MSFT_s011
                                 + 1.10423 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.896904 w_k1_ADM_s011
                                 + 0.660041 w_k1_AXP_s011
                                 + 0.611684 w_k1_DIS_s011
                                 + 0.874411 w_k1_FDX_s011
                                 + 0.500512 w_k1_IBM_s011
                                 + 1.02947 w_k1_INTC_s011
                                 + 0.59543 w_k1_KO_s011
                                 + 0.367861 w_k1_MAT_s011
                                 + 0.989756 w_k1_MSFT_s011
                                 + 0.769843 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.0108 w_k1_ADM_s011
                                 + 0.817913 w_k1_AXP_s011
                                 + 0.882941 w_k1_DIS_s011
                                 + 0.972938 w_k1_FDX_s011
                                 + 0.736569 w_k1_IBM_s011
                                 + 1.01987 w_k1_INTC_s011
                                 + 0.920936 w_k1_KO_s011
                                 + 0.571226 w_k1_MAT_s011
                                 + 1.02308 w_k1_MSFT_s011
                                 + 0.969663 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.42592 w_k1_ADM_s011
                                 + 1.42077 w_k1_AXP_s011 + 1.3251 w_k1_DIS_s011
                                 + 1.27108 w_k1_FDX_s011
                                 + 1.38615 w_k1_IBM_s011
                                 + 1.28936 w_k1_INTC_s011
                                 + 1.38228 w_k1_KO_s011 + 1.69841 w_k1_MAT_s011
                                 + 1.32222 w_k1_MSFT_s011
                                 + 1.24992 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.840367 w_k1_ADM_s011
                                 + 0.768494 w_k1_AXP_s011
                                 + 0.738803 w_k1_DIS_s011
                                 + 0.749733 w_k1_FDX_s011
                                 + 0.915955 w_k1_IBM_s011
                                 + 0.879062 w_k1_INTC_s011
                                 + 0.86986 w_k1_KO_s011
                                 + 0.933406 w_k1_MAT_s011
                                 + 0.874738 w_k1_MSFT_s011
                                 + 0.938855 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.13785 w_k1_ADM_s011
                                 + 0.878903 w_k1_AXP_s011
                                 + 0.87701 w_k1_DIS_s011
                                 + 0.93438 w_k1_FDX_s011
                                 + 1.16078 w_k1_IBM_s011
                                 + 1.07971 w_k1_INTC_s011
                                 + 1.04062 w_k1_KO_s011
                                 + 0.804781 w_k1_MAT_s011
                                 + 1.00025 w_k1_MSFT_s011 + 1.044 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.13985 w_k1_ADM_s011
                                 + 0.926758 w_k1_AXP_s011
                                 + 1.33245 w_k1_DIS_s011
                                 + 1.22516 w_k1_FDX_s011
                                 + 0.850881 w_k1_IBM_s011
                                 + 1.24222 w_k1_INTC_s011 + 1.0363 w_k1_KO_s011
                                 + 0.725691 w_k1_MAT_s011
                                 + 1.29066 w_k1_MSFT_s011
                                 + 0.992742 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.06499 w_k1_ADM_s011
                                 + 1.07557 w_k1_AXP_s011
                                 + 0.979032 w_k1_DIS_s011
                                 + 0.880554 w_k1_FDX_s011
                                 + 1.80565 w_k1_IBM_s011
                                 + 0.990666 w_k1_INTC_s011
                                 + 1.18206 w_k1_KO_s011 + 1.41117 w_k1_MAT_s011
                                 + 0.999752 w_k1_MSFT_s011
                                 + 1.06359 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.747092 w_k1_ADM_s012
                                 + 1.09326 w_k1_AXP_s012
                                 + 1.13586 w_k1_DIS_s012
                                 + 0.902236 w_k1_FDX_s012
                                 + 1.89826 w_k1_IBM_s012
                                 + 0.924931 w_k1_INTC_s012
                                 + 1.0889 w_k1_KO_s012 + 1.92413 w_k1_MAT_s012
                                 + 0.903545 w_k1_MSFT_s012
                                 + 1.11203 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.35898 w_k1_ADM_s012
                                 + 1.67915 w_k1_AXP_s012
                                 + 1.64655 w_k1_DIS_s012 + 1.286 w_k1_FDX_s012
                                 + 1.65881 w_k1_IBM_s012
                                 + 1.21395 w_k1_INTC_s012
                                 + 1.45516 w_k1_KO_s012 + 2.17527 w_k1_MAT_s012
                                 + 1.19821 w_k1_MSFT_s012
                                 + 1.26905 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.897009 w_k1_ADM_s012
                                 + 1.02795 w_k1_AXP_s012
                                 + 0.895396 w_k1_DIS_s012
                                 + 0.853132 w_k1_FDX_s012
                                 + 1.36768 w_k1_IBM_s012
                                 + 0.981238 w_k1_INTC_s012
                                 + 1.08193 w_k1_KO_s012 + 1.55397 w_k1_MAT_s012
                                 + 0.945589 w_k1_MSFT_s012
                                 + 1.07952 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.3101 w_k1_ADM_s012
                                 + 1.25858 w_k1_AXP_s012
                                 + 0.982058 w_k1_DIS_s012
                                 + 1.24013 w_k1_FDX_s012
                                 + 0.780576 w_k1_IBM_s012
                                 + 1.25064 w_k1_INTC_s012
                                 + 0.815605 w_k1_KO_s012
                                 + 1.05749 w_k1_MAT_s012
                                 + 0.996807 w_k1_MSFT_s012 + 1.05 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.88904 w_k1_ADM_s012
                                 + 1.04567 w_k1_AXP_s012
                                 + 0.885906 w_k1_DIS_s012
                                 + 0.984057 w_k1_FDX_s012
                                 + 0.820495 w_k1_IBM_s012
                                 + 1.14915 w_k1_INTC_s012
                                 + 0.795469 w_k1_KO_s012
                                 + 0.797517 w_k1_MAT_s012
                                 + 0.893942 w_k1_MSFT_s012
                                 + 0.994752 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.22274 w_k1_ADM_s012
                                 + 1.05016 w_k1_AXP_s012
                                 + 1.33566 w_k1_DIS_s012
                                 + 1.09826 w_k1_FDX_s012 + 1.2339 w_k1_IBM_s012
                                 + 1.06496 w_k1_INTC_s012
                                 + 1.23852 w_k1_KO_s012 + 1.17917 w_k1_MAT_s012
                                 + 1.08432 w_k1_MSFT_s012
                                 + 1.09953 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.972441 w_k1_ADM_s012
                                 + 1.01951 w_k1_AXP_s012
                                 + 1.26397 w_k1_DIS_s012
                                 + 1.22084 w_k1_FDX_s012
                                 + 0.862686 w_k1_IBM_s012
                                 + 1.27123 w_k1_INTC_s012
                                 + 1.05466 w_k1_KO_s012
                                 + 0.873663 w_k1_MAT_s012
                                 + 1.12071 w_k1_MSFT_s012
                                 + 1.18939 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.883713 w_k1_ADM_s012
                                 + 0.743359 w_k1_AXP_s012
                                 + 0.666927 w_k1_DIS_s012
                                 + 0.811465 w_k1_FDX_s012
                                 + 0.793258 w_k1_IBM_s012
                                 + 0.92186 w_k1_INTC_s012
                                 + 0.784075 w_k1_KO_s012
                                 + 0.806852 w_k1_MAT_s012
                                 + 0.815356 w_k1_MSFT_s012
                                 + 0.834002 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.66466 w_k1_ADM_s013
                                 + 0.65185 w_k1_AXP_s013
                                 + 0.716847 w_k1_DIS_s013
                                 + 0.684325 w_k1_FDX_s013
                                 + 0.912274 w_k1_IBM_s013
                                 + 0.840754 w_k1_INTC_s013
                                 + 0.824047 w_k1_KO_s013
                                 + 0.690797 w_k1_MAT_s013
                                 + 0.853126 w_k1_MSFT_s013
                                 + 0.788876 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.834302 w_k1_ADM_s013
                                 + 0.821241 w_k1_AXP_s013
                                 + 0.743357 w_k1_DIS_s013
                                 + 0.817715 w_k1_FDX_s013
                                 + 0.90573 w_k1_IBM_s013
                                 + 0.906195 w_k1_INTC_s013
                                 + 0.774493 w_k1_KO_s013
                                 + 0.710807 w_k1_MAT_s013
                                 + 0.815242 w_k1_MSFT_s013
                                 + 0.859894 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.729056 w_k1_ADM_s013
                                 + 1.17042 w_k1_AXP_s013
                                 + 1.17546 w_k1_DIS_s013
                                 + 0.956051 w_k1_FDX_s013
                                 + 1.87697 w_k1_IBM_s013
                                 + 1.20152 w_k1_INTC_s013
                                 + 1.19939 w_k1_KO_s013 + 1.2799 w_k1_MAT_s013
                                 + 1.05737 w_k1_MSFT_s013
                                 + 0.954641 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.32168 w_k1_ADM_s013
                                 + 1.79741 w_k1_AXP_s013
                                 + 1.92291 w_k1_DIS_s013 + 1.5581 w_k1_FDX_s013
                                 + 1.36008 w_k1_IBM_s013
                                 + 1.30876 w_k1_INTC_s013
                                 + 1.24728 w_k1_KO_s013 + 1.3001 w_k1_MAT_s013
                                 + 1.30399 w_k1_MSFT_s013
                                 + 1.12388 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.809755 w_k1_ADM_s013
                                 + 0.886154 w_k1_AXP_s013
                                 + 0.979749 w_k1_DIS_s013 + 1.056 w_k1_FDX_s013
                                 + 0.647968 w_k1_IBM_s013
                                 + 1.11787 w_k1_INTC_s013
                                 + 0.764541 w_k1_KO_s013
                                 + 0.545448 w_k1_MAT_s013
                                 + 1.0497 w_k1_MSFT_s013
                                 + 0.830744 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.930519 w_k1_ADM_s013
                                 + 0.982884 w_k1_AXP_s013
                                 + 1.02941 w_k1_DIS_s013
                                 + 0.989026 w_k1_FDX_s013
                                 + 1.13919 w_k1_IBM_s013
                                 + 1.01092 w_k1_INTC_s013
                                 + 0.847954 w_k1_KO_s013
                                 + 0.984825 w_k1_MAT_s013
                                 + 0.921046 w_k1_MSFT_s013
                                 + 0.959516 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.15917 w_k1_ADM_s013
                                 + 1.15444 w_k1_AXP_s013
                                 + 1.30431 w_k1_DIS_s013
                                 + 1.27826 w_k1_FDX_s013
                                 + 0.765384 w_k1_IBM_s013
                                 + 1.00321 w_k1_INTC_s013
                                 + 0.893894 w_k1_KO_s013
                                 + 0.738418 w_k1_MAT_s013
                                 + 1.11726 w_k1_MSFT_s013
                                 + 0.989697 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.01663 w_k1_ADM_s013
                                 + 1.1943 w_k1_AXP_s013 + 1.46158 w_k1_DIS_s013
                                 + 1.23641 w_k1_FDX_s013 + 1.3026 w_k1_IBM_s013
                                 + 0.978669 w_k1_INTC_s013
                                 + 1.34867 w_k1_KO_s013 + 1.16501 w_k1_MAT_s013
                                 + 1.06371 w_k1_MSFT_s013
                                 + 1.14692 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.731478 w_k1_ADM_s014
                                 + 0.492658 w_k1_AXP_s014
                                 + 0.549753 w_k1_DIS_s014
                                 + 0.678724 w_k1_FDX_s014
                                 + 0.688643 w_k1_IBM_s014
                                 + 0.678218 w_k1_INTC_s014
                                 + 0.661774 w_k1_KO_s014
                                 + 0.501603 w_k1_MAT_s014
                                 + 0.70587 w_k1_MSFT_s014
                                 + 0.728375 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.22081 w_k1_ADM_s014
                                 + 1.09686 w_k1_AXP_s014
                                 + 1.14766 w_k1_DIS_s014
                                 + 1.19998 w_k1_FDX_s014
                                 + 0.804133 w_k1_IBM_s014
                                 + 1.06671 w_k1_INTC_s014
                                 + 0.975228 w_k1_KO_s014
                                 + 0.774394 w_k1_MAT_s014
                                 + 1.07383 w_k1_MSFT_s014
                                 + 1.06832 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.770744 w_k1_ADM_s014
                                 + 0.79413 w_k1_AXP_s014
                                 + 0.826825 w_k1_DIS_s014
                                 + 0.873629 w_k1_FDX_s014
                                 + 0.811351 w_k1_IBM_s014
                                 + 0.905167 w_k1_INTC_s014
                                 + 0.873231 w_k1_KO_s014 + 1.0327 w_k1_MAT_s014
                                 + 0.815422 w_k1_MSFT_s014
                                 + 0.95751 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.15134 w_k1_ADM_s014
                                 + 1.42923 w_k1_AXP_s014
                                 + 1.38922 w_k1_DIS_s014
                                 + 1.47911 w_k1_FDX_s014
                                 + 0.852515 w_k1_IBM_s014
                                 + 1.49247 w_k1_INTC_s014
                                 + 1.14237 w_k1_KO_s014 + 1.42634 w_k1_MAT_s014
                                 + 1.17273 w_k1_MSFT_s014
                                 + 1.27164 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.27158 w_k1_ADM_s014
                                 + 1.57103 w_k1_AXP_s014
                                 + 1.32025 w_k1_DIS_s014
                                 + 1.28554 w_k1_FDX_s014
                                 + 1.18918 w_k1_IBM_s014
                                 + 1.44416 w_k1_INTC_s014
                                 + 1.18149 w_k1_KO_s014 + 1.08536 w_k1_MAT_s014
                                 + 1.11536 w_k1_MSFT_s014
                                 + 1.13183 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.866477 w_k1_ADM_s014
                                 + 0.866128 w_k1_AXP_s014
                                 + 0.687496 w_k1_DIS_s014
                                 + 0.862551 w_k1_FDX_s014
                                 + 0.923845 w_k1_IBM_s014
                                 + 0.925138 w_k1_INTC_s014
                                 + 0.728032 w_k1_KO_s014
                                 + 0.655056 w_k1_MAT_s014
                                 + 0.782127 w_k1_MSFT_s014
                                 + 0.869252 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.08731 w_k1_ADM_s014
                                 + 1.05998 w_k1_AXP_s014
                                 + 0.96605 w_k1_DIS_s014
                                 + 1.04788 w_k1_FDX_s014
                                 + 1.45812 w_k1_IBM_s014
                                 + 0.909138 w_k1_INTC_s014
                                 + 1.23537 w_k1_KO_s014 + 1.60949 w_k1_MAT_s014
                                 + 0.957269 w_k1_MSFT_s014
                                 + 1.14453 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.900069 w_k1_ADM_s014
                                 + 0.909094 w_k1_AXP_s014
                                 + 0.939105 w_k1_DIS_s014
                                 + 0.930088 w_k1_FDX_s014 + 1.113 w_k1_IBM_s014
                                 + 0.940459 w_k1_INTC_s014
                                 + 1.01103 w_k1_KO_s014 + 1.02133 w_k1_MAT_s014
                                 + 0.997714 w_k1_MSFT_s014
                                 + 0.943546 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.06435 w_k1_ADM_s015
                                 + 0.711346 w_k1_AXP_s015
                                 + 1.1097 w_k1_DIS_s015 + 1.03773 w_k1_FDX_s015
                                 + 0.888752 w_k1_IBM_s015
                                 + 0.911345 w_k1_INTC_s015
                                 + 1.14629 w_k1_KO_s015
                                 + 0.898658 w_k1_MAT_s015
                                 + 1.0389 w_k1_MSFT_s015 + 1.1042 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.35979 w_k1_ADM_s015
                                 + 1.15185 w_k1_AXP_s015
                                 + 1.50233 w_k1_DIS_s015
                                 + 1.42847 w_k1_FDX_s015
                                 + 0.746397 w_k1_IBM_s015
                                 + 1.25057 w_k1_INTC_s015
                                 + 1.11571 w_k1_KO_s015
                                 + 0.641721 w_k1_MAT_s015
                                 + 1.33534 w_k1_MSFT_s015
                                 + 1.23318 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.880835 w_k1_ADM_s015
                                 + 0.609077 w_k1_AXP_s015
                                 + 0.672149 w_k1_DIS_s015
                                 + 0.763549 w_k1_FDX_s015
                                 + 0.837683 w_k1_IBM_s015
                                 + 0.990728 w_k1_INTC_s015
                                 + 0.97995 w_k1_KO_s015
                                 + 0.795362 w_k1_MAT_s015
                                 + 0.85723 w_k1_MSFT_s015
                                 + 0.953295 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.50067 w_k1_ADM_s015
                                 + 1.27104 w_k1_AXP_s015
                                 + 1.53253 w_k1_DIS_s015
                                 + 1.32185 w_k1_FDX_s015
                                 + 1.31952 w_k1_IBM_s015
                                 + 1.15219 w_k1_INTC_s015
                                 + 1.32707 w_k1_KO_s015 + 1.69255 w_k1_MAT_s015
                                 + 1.18919 w_k1_MSFT_s015
                                 + 1.38965 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.1052 w_k1_ADM_s015
                                 + 1.01762 w_k1_AXP_s015
                                 + 1.05358 w_k1_DIS_s015
                                 + 1.14702 w_k1_FDX_s015
                                 + 0.798116 w_k1_IBM_s015
                                 + 1.15918 w_k1_INTC_s015
                                 + 0.934347 w_k1_KO_s015
                                 + 0.556714 w_k1_MAT_s015
                                 + 1.08574 w_k1_MSFT_s015
                                 + 1.04132 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.668371 w_k1_ADM_s015
                                 + 0.754546 w_k1_AXP_s015
                                 + 0.945435 w_k1_DIS_s015
                                 + 0.820084 w_k1_FDX_s015
                                 + 1.51049 w_k1_IBM_s015
                                 + 1.06709 w_k1_INTC_s015
                                 + 1.19024 w_k1_KO_s015 + 1.45404 w_k1_MAT_s015
                                 + 0.937016 w_k1_MSFT_s015
                                 + 1.14167 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.935924 w_k1_ADM_s015
                                 + 0.914228 w_k1_AXP_s015
                                 + 1.38286 w_k1_DIS_s015
                                 + 1.11023 w_k1_FDX_s015
                                 + 1.28515 w_k1_IBM_s015
                                 + 1.19242 w_k1_INTC_s015
                                 + 1.41808 w_k1_KO_s015
                                 + 0.868344 w_k1_MAT_s015
                                 + 1.18396 w_k1_MSFT_s015
                                 + 1.20889 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01205 w_k1_ADM_s015
                                 + 1.08677 w_k1_AXP_s015
                                 + 0.964125 w_k1_DIS_s015
                                 + 0.994234 w_k1_FDX_s015
                                 + 1.15544 w_k1_IBM_s015
                                 + 1.11407 w_k1_INTC_s015
                                 + 1.26331 w_k1_KO_s015 + 1.17281 w_k1_MAT_s015
                                 + 1.0013 w_k1_MSFT_s015 + 1.02859 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.2293 w_k1_ADM_s016
                                 + 1.18811 w_k1_AXP_s016
                                 + 1.17129 w_k1_DIS_s016
                                 + 1.13894 w_k1_FDX_s016
                                 + 1.13129 w_k1_IBM_s016
                                 + 1.10858 w_k1_INTC_s016
                                 + 1.23343 w_k1_KO_s016 + 1.01103 w_k1_MAT_s016
                                 + 1.20287 w_k1_MSFT_s016
                                 + 1.03228 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.799615 w_k1_ADM_s016
                                 + 0.551515 w_k1_AXP_s016
                                 + 0.577841 w_k1_DIS_s016
                                 + 0.653401 w_k1_FDX_s016
                                 + 0.781605 w_k1_IBM_s016
                                 + 0.727696 w_k1_INTC_s016
                                 + 0.823388 w_k1_KO_s016
                                 + 0.613731 w_k1_MAT_s016
                                 + 0.759569 w_k1_MSFT_s016
                                 + 0.751577 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.10981 w_k1_ADM_s016
                                 + 1.12197 w_k1_AXP_s016
                                 + 0.986251 w_k1_DIS_s016
                                 + 0.974478 w_k1_FDX_s016
                                 + 1.13751 w_k1_IBM_s016
                                 + 1.12902 w_k1_INTC_s016
                                 + 1.18417 w_k1_KO_s016 + 1.50503 w_k1_MAT_s016
                                 + 1.06597 w_k1_MSFT_s016
                                 + 1.12458 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.891809 w_k1_ADM_s016
                                 + 0.714443 w_k1_AXP_s016
                                 + 0.667315 w_k1_DIS_s016 + 0.814 w_k1_FDX_s016
                                 + 0.716495 w_k1_IBM_s016
                                 + 0.876277 w_k1_INTC_s016
                                 + 0.801183 w_k1_KO_s016
                                 + 0.724581 w_k1_MAT_s016
                                 + 0.873137 w_k1_MSFT_s016
                                 + 0.876454 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.23891 w_k1_ADM_s016
                                 + 1.19124 w_k1_AXP_s016
                                 + 1.05432 w_k1_DIS_s016
                                 + 0.975591 w_k1_FDX_s016
                                 + 1.62149 w_k1_IBM_s016
                                 + 1.16319 w_k1_INTC_s016
                                 + 1.44164 w_k1_KO_s016 + 1.85756 w_k1_MAT_s016
                                 + 1.14184 w_k1_MSFT_s016
                                 + 1.18271 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.0815 w_k1_ADM_s016
                                 + 1.24681 w_k1_AXP_s016
                                 + 0.758801 w_k1_DIS_s016
                                 + 0.960949 w_k1_FDX_s016
                                 + 0.878504 w_k1_IBM_s016
                                 + 0.907762 w_k1_INTC_s016
                                 + 0.98893 w_k1_KO_s016 + 1.4609 w_k1_MAT_s016
                                 + 0.871265 w_k1_MSFT_s016
                                 + 1.00331 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.19853 w_k1_ADM_s016
                                 + 1.06701 w_k1_AXP_s016
                                 + 1.16747 w_k1_DIS_s016
                                 + 1.19633 w_k1_FDX_s016
                                 + 0.834526 w_k1_IBM_s016
                                 + 1.09198 w_k1_INTC_s016
                                 + 0.92126 w_k1_KO_s016
                                 + 0.802627 w_k1_MAT_s016
                                 + 1.16247 w_k1_MSFT_s016
                                 + 1.07112 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.793613 w_k1_ADM_s016
                                 + 0.693794 w_k1_AXP_s016
                                 + 0.692659 w_k1_DIS_s016
                                 + 0.732623 w_k1_FDX_s016
                                 + 1.15978 w_k1_IBM_s016
                                 + 0.868191 w_k1_INTC_s016
                                 + 1.02567 w_k1_KO_s016 + 1.29752 w_k1_MAT_s016
                                 + 0.830574 w_k1_MSFT_s016
                                 + 0.885877 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.938252 w_k1_ADM_s017
                                 + 0.891403 w_k1_AXP_s017
                                 + 0.860025 w_k1_DIS_s017
                                 + 0.995824 w_k1_FDX_s017
                                 + 0.76837 w_k1_IBM_s017
                                 + 0.861327 w_k1_INTC_s017
                                 + 0.749602 w_k1_KO_s017
                                 + 0.74776 w_k1_MAT_s017
                                 + 0.961438 w_k1_MSFT_s017
                                 + 0.960752 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.968154 w_k1_ADM_s017
                                 + 0.905041 w_k1_AXP_s017
                                 + 0.901493 w_k1_DIS_s017
                                 + 0.936307 w_k1_FDX_s017
                                 + 1.24274 w_k1_IBM_s017
                                 + 1.00043 w_k1_INTC_s017
                                 + 1.15065 w_k1_KO_s017
                                 + 0.955709 w_k1_MAT_s017
                                 + 1.06684 w_k1_MSFT_s017
                                 + 1.04961 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.23155 w_k1_ADM_s017
                                 + 1.32812 w_k1_AXP_s017
                                 + 1.42927 w_k1_DIS_s017
                                 + 1.31856 w_k1_FDX_s017 + 0.8864 w_k1_IBM_s017
                                 + 1.25625 w_k1_INTC_s017
                                 + 1.16061 w_k1_KO_s017
                                 + 0.871287 w_k1_MAT_s017
                                 + 1.40855 w_k1_MSFT_s017
                                 + 1.18635 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00465 w_k1_ADM_s017
                                 + 0.996496 w_k1_AXP_s017
                                 + 1.60962 w_k1_DIS_s017
                                 + 1.14684 w_k1_FDX_s017
                                 + 1.36276 w_k1_IBM_s017
                                 + 0.999761 w_k1_INTC_s017
                                 + 1.36714 w_k1_KO_s017 + 1.09408 w_k1_MAT_s017
                                 + 1.23076 w_k1_MSFT_s017
                                 + 1.14443 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.35196 w_k1_ADM_s017
                                 + 1.30574 w_k1_AXP_s017
                                 + 0.997834 w_k1_DIS_s017
                                 + 1.26147 w_k1_FDX_s017
                                 + 0.674001 w_k1_IBM_s017
                                 + 1.173 w_k1_INTC_s017 + 1.02756 w_k1_KO_s017
                                 + 0.75858 w_k1_MAT_s017
                                 + 1.28006 w_k1_MSFT_s017
                                 + 0.989983 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.745297 w_k1_ADM_s017
                                 + 0.69891 w_k1_AXP_s017
                                 + 0.673577 w_k1_DIS_s017
                                 + 0.739028 w_k1_FDX_s017
                                 + 0.834404 w_k1_IBM_s017
                                 + 0.813647 w_k1_INTC_s017
                                 + 0.793033 w_k1_KO_s017
                                 + 0.618538 w_k1_MAT_s017
                                 + 0.8467 w_k1_MSFT_s017
                                 + 0.789258 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.11128 w_k1_ADM_s017
                                 + 1.29238 w_k1_AXP_s017 + 1.1067 w_k1_DIS_s017
                                 + 1.18629 w_k1_FDX_s017
                                 + 1.08513 w_k1_IBM_s017
                                 + 1.10376 w_k1_INTC_s017
                                 + 1.11558 w_k1_KO_s017 + 1.30086 w_k1_MAT_s017
                                 + 1.19782 w_k1_MSFT_s017
                                 + 1.26701 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.2198 w_k1_ADM_s017
                                 + 1.2301 w_k1_AXP_s017 + 1.28673 w_k1_DIS_s017
                                 + 1.10606 w_k1_FDX_s017
                                 + 1.71874 w_k1_IBM_s017
                                 + 0.896669 w_k1_INTC_s017
                                 + 1.34499 w_k1_KO_s017 + 1.58233 w_k1_MAT_s017
                                 + 1.16664 w_k1_MSFT_s017
                                 + 1.05174 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.7323 w_k1_ADM_s100
                                 + 1.30765 w_k1_AXP_s100
                                 + 1.11528 w_k1_DIS_s100
                                 + 1.26677 w_k1_FDX_s100
                                 + 0.892163 w_k1_IBM_s100
                                 + 1.19485 w_k1_INTC_s100
                                 + 0.994756 w_k1_KO_s100
                                 + 0.77111 w_k1_MAT_s100
                                 + 1.00023 w_k1_MSFT_s100
                                 + 1.02527 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14997 w_k1_ADM_s100
                                 + 0.928007 w_k1_AXP_s100
                                 + 0.73681 w_k1_DIS_s100
                                 + 0.892788 w_k1_FDX_s100
                                 + 1.01787 w_k1_IBM_s100
                                 + 1.03384 w_k1_INTC_s100
                                 + 0.934697 w_k1_KO_s100
                                 + 0.684593 w_k1_MAT_s100
                                 + 0.754864 w_k1_MSFT_s100
                                 + 0.849259 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.41296 w_k1_ADM_s100
                                 + 1.39362 w_k1_AXP_s100 + 1.554 w_k1_DIS_s100
                                 + 1.14887 w_k1_FDX_s100
                                 + 1.65983 w_k1_IBM_s100
                                 + 1.00648 w_k1_INTC_s100 + 1.4325 w_k1_KO_s100
                                 + 1.69747 w_k1_MAT_s100
                                 + 1.19533 w_k1_MSFT_s100
                                 + 1.32408 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.912577 w_k1_ADM_s100
                                 + 1.04467 w_k1_AXP_s100
                                 + 0.929967 w_k1_DIS_s100
                                 + 0.916196 w_k1_FDX_s100
                                 + 1.28351 w_k1_IBM_s100
                                 + 0.915975 w_k1_INTC_s100
                                 + 0.989035 w_k1_KO_s100
                                 + 1.13043 w_k1_MAT_s100
                                 + 0.884044 w_k1_MSFT_s100
                                 + 0.896133 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.00635 w_k1_ADM_s100
                                 + 0.582611 w_k1_AXP_s100
                                 + 0.58941 w_k1_DIS_s100
                                 + 0.705493 w_k1_FDX_s100
                                 + 0.77343 w_k1_IBM_s100
                                 + 0.951978 w_k1_INTC_s100
                                 + 0.771602 w_k1_KO_s100
                                 + 0.457962 w_k1_MAT_s100
                                 + 0.704626 w_k1_MSFT_s100
                                 + 0.860315 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.86521 w_k1_ADM_s100
                                 + 0.656805 w_k1_AXP_s100
                                 + 0.720014 w_k1_DIS_s100
                                 + 0.807661 w_k1_FDX_s100
                                 + 0.616779 w_k1_IBM_s100
                                 + 0.924229 w_k1_INTC_s100
                                 + 0.78554 w_k1_KO_s100
                                 + 0.486385 w_k1_MAT_s100
                                 + 0.914616 w_k1_MSFT_s100
                                 + 0.848878 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.12838 w_k1_ADM_s100
                                 + 1.05004 w_k1_AXP_s100
                                 + 1.19138 w_k1_DIS_s100
                                 + 1.05986 w_k1_FDX_s100
                                 + 1.09564 w_k1_IBM_s100
                                 + 1.35782 w_k1_INTC_s100
                                 + 1.17179 w_k1_KO_s100
                                 + 0.854042 w_k1_MAT_s100
                                 + 1.16207 w_k1_MSFT_s100 + 1.0383 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.07255 w_k1_ADM_s100
                                 + 0.771613 w_k1_AXP_s100
                                 + 0.761144 w_k1_DIS_s100
                                 + 0.932245 w_k1_FDX_s100
                                 + 0.738065 w_k1_IBM_s100
                                 + 0.864781 w_k1_INTC_s100
                                 + 0.662723 w_k1_KO_s100
                                 + 0.569563 w_k1_MAT_s100
                                 + 0.849369 w_k1_MSFT_s100
                                 + 0.81246 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.821768 w_k1_ADM_s101
                                 + 0.746155 w_k1_AXP_s101
                                 + 0.691462 w_k1_DIS_s101
                                 + 0.983241 w_k1_FDX_s101
                                 + 0.602067 w_k1_IBM_s101
                                 + 0.941754 w_k1_INTC_s101
                                 + 0.668045 w_k1_KO_s101
                                 + 0.402832 w_k1_MAT_s101
                                 + 0.827576 w_k1_MSFT_s101
                                 + 0.773917 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.765346 w_k1_ADM_s101
                                 + 0.666042 w_k1_AXP_s101
                                 + 0.952738 w_k1_DIS_s101
                                 + 0.867967 w_k1_FDX_s101
                                 + 0.742876 w_k1_IBM_s101
                                 + 0.850561 w_k1_INTC_s101 + 1.009 w_k1_KO_s101
                                 + 0.515156 w_k1_MAT_s101
                                 + 0.943454 w_k1_MSFT_s101
                                 + 0.986427 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.05577 w_k1_ADM_s101
                                 + 1.23982 w_k1_AXP_s101
                                 + 0.663042 w_k1_DIS_s101
                                 + 0.92055 w_k1_FDX_s101
                                 + 1.00199 w_k1_IBM_s101
                                 + 0.882503 w_k1_INTC_s101
                                 + 0.869675 w_k1_KO_s101
                                 + 1.30617 w_k1_MAT_s101
                                 + 0.785067 w_k1_MSFT_s101
                                 + 0.903759 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.788118 w_k1_ADM_s101
                                 + 0.755048 w_k1_AXP_s101
                                 + 0.976714 w_k1_DIS_s101
                                 + 0.959905 w_k1_FDX_s101
                                 + 0.856709 w_k1_IBM_s101
                                 + 1.05756 w_k1_INTC_s101
                                 + 0.960113 w_k1_KO_s101
                                 + 0.646476 w_k1_MAT_s101
                                 + 0.897708 w_k1_MSFT_s101
                                 + 0.948743 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.02756 w_k1_ADM_s101
                                 + 1.1667 w_k1_AXP_s101 + 1.83434 w_k1_DIS_s101
                                 + 1.35229 w_k1_FDX_s101
                                 + 1.70543 w_k1_IBM_s101
                                 + 1.09987 w_k1_INTC_s101
                                 + 1.57357 w_k1_KO_s101 + 1.46866 w_k1_MAT_s101
                                 + 1.08681 w_k1_MSFT_s101
                                 + 1.28356 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.0015 w_k1_ADM_s101
                                 + 1.22894 w_k1_AXP_s101
                                 + 1.00239 w_k1_DIS_s101
                                 + 1.01249 w_k1_FDX_s101
                                 + 1.12232 w_k1_IBM_s101
                                 + 0.825165 w_k1_INTC_s101
                                 + 0.987822 w_k1_KO_s101
                                 + 1.03864 w_k1_MAT_s101
                                 + 0.863617 w_k1_MSFT_s101
                                 + 1.00283 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.989103 w_k1_ADM_s101
                                 + 1.33214 w_k1_AXP_s101
                                 + 1.17309 w_k1_DIS_s101
                                 + 1.16967 w_k1_FDX_s101
                                 + 1.18111 w_k1_IBM_s101
                                 + 1.13031 w_k1_INTC_s101
                                 + 1.12612 w_k1_KO_s101 + 1.04906 w_k1_MAT_s101
                                 + 0.955657 w_k1_MSFT_s101
                                 + 1.09434 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.912125 w_k1_ADM_s101
                                 + 0.85952 w_k1_AXP_s101
                                 + 0.760178 w_k1_DIS_s101
                                 + 0.826946 w_k1_FDX_s101
                                 + 1.12062 w_k1_IBM_s101
                                 + 0.93167 w_k1_INTC_s101
                                 + 1.06275 w_k1_KO_s101
                                 + 0.896027 w_k1_MAT_s101
                                 + 0.784062 w_k1_MSFT_s101
                                 + 0.924464 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.24857 w_k1_ADM_s102
                                 + 1.18538 w_k1_AXP_s102
                                 + 1.63674 w_k1_DIS_s102
                                 + 1.24606 w_k1_FDX_s102
                                 + 1.24497 w_k1_IBM_s102
                                 + 1.29292 w_k1_INTC_s102
                                 + 1.37993 w_k1_KO_s102 + 1.18208 w_k1_MAT_s102
                                 + 1.45949 w_k1_MSFT_s102
                                 + 1.22397 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.878526 w_k1_ADM_s102
                                 + 0.622682 w_k1_AXP_s102
                                 + 0.751551 w_k1_DIS_s102
                                 + 0.8263 w_k1_FDX_s102
                                 + 0.636547 w_k1_IBM_s102
                                 + 1.03885 w_k1_INTC_s102
                                 + 0.649732 w_k1_KO_s102
                                 + 0.447494 w_k1_MAT_s102
                                 + 0.903955 w_k1_MSFT_s102
                                 + 0.75715 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.926591 w_k1_ADM_s102
                                 + 0.818195 w_k1_AXP_s102
                                 + 1.03787 w_k1_DIS_s102
                                 + 0.933722 w_k1_FDX_s102
                                 + 0.885685 w_k1_IBM_s102
                                 + 1.36121 w_k1_INTC_s102
                                 + 0.908805 w_k1_KO_s102
                                 + 0.590739 w_k1_MAT_s102
                                 + 0.992436 w_k1_MSFT_s102
                                 + 0.898511 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.928547 w_k1_ADM_s102
                                 + 0.838585 w_k1_AXP_s102
                                 + 1.03909 w_k1_DIS_s102
                                 + 1.10919 w_k1_FDX_s102
                                 + 0.791228 w_k1_IBM_s102
                                 + 1.35809 w_k1_INTC_s102
                                 + 0.908183 w_k1_KO_s102
                                 + 0.78254 w_k1_MAT_s102
                                 + 1.09438 w_k1_MSFT_s102
                                 + 1.02092 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.939273 w_k1_ADM_s102
                                 + 0.724206 w_k1_AXP_s102
                                 + 0.760972 w_k1_DIS_s102
                                 + 0.916022 w_k1_FDX_s102
                                 + 1.16667 w_k1_IBM_s102
                                 + 1.03929 w_k1_INTC_s102
                                 + 0.96927 w_k1_KO_s102
                                 + 0.898298 w_k1_MAT_s102
                                 + 0.951745 w_k1_MSFT_s102
                                 + 0.997518 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.21551 w_k1_ADM_s102
                                 + 1.17762 w_k1_AXP_s102
                                 + 1.37131 w_k1_DIS_s102
                                 + 1.08234 w_k1_FDX_s102 + 1.6074 w_k1_IBM_s102
                                 + 1.16374 w_k1_INTC_s102
                                 + 1.27665 w_k1_KO_s102 + 1.24353 w_k1_MAT_s102
                                 + 1.104 w_k1_MSFT_s102 + 1.12585 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.38483 w_k1_ADM_s102
                                 + 1.38352 w_k1_AXP_s102
                                 + 1.25947 w_k1_DIS_s102 + 1.2874 w_k1_FDX_s102
                                 + 1.06892 w_k1_IBM_s102
                                 + 1.53205 w_k1_INTC_s102
                                 + 1.21702 w_k1_KO_s102 + 1.65046 w_k1_MAT_s102
                                 + 1.34976 w_k1_MSFT_s102
                                 + 1.24889 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.848161 w_k1_ADM_s102
                                 + 0.568794 w_k1_AXP_s102
                                 + 0.713038 w_k1_DIS_s102
                                 + 0.780871 w_k1_FDX_s102
                                 + 0.90557 w_k1_IBM_s102
                                 + 0.998828 w_k1_INTC_s102
                                 + 0.77574 w_k1_KO_s102
                                 + 0.567649 w_k1_MAT_s102
                                 + 0.754208 w_k1_MSFT_s102
                                 + 0.872714 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.786135 w_k1_ADM_s103
                                 + 0.521523 w_k1_AXP_s103
                                 + 0.538485 w_k1_DIS_s103
                                 + 0.689078 w_k1_FDX_s103
                                 + 0.723397 w_k1_IBM_s103
                                 + 0.827904 w_k1_INTC_s103
                                 + 0.614314 w_k1_KO_s103
                                 + 0.528492 w_k1_MAT_s103
                                 + 0.727708 w_k1_MSFT_s103
                                 + 0.804595 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.907614 w_k1_ADM_s103
                                 + 0.699766 w_k1_AXP_s103
                                 + 0.893379 w_k1_DIS_s103
                                 + 0.892313 w_k1_FDX_s103
                                 + 0.789988 w_k1_IBM_s103
                                 + 0.790376 w_k1_INTC_s103
                                 + 0.803272 w_k1_KO_s103
                                 + 0.717591 w_k1_MAT_s103
                                 + 0.990077 w_k1_MSFT_s103
                                 + 0.890649 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.0446 w_k1_ADM_s103
                                 + 1.18108 w_k1_AXP_s103
                                 + 1.35071 w_k1_DIS_s103
                                 + 1.24822 w_k1_FDX_s103
                                 + 0.986671 w_k1_IBM_s103
                                 + 1.43676 w_k1_INTC_s103
                                 + 1.06182 w_k1_KO_s103 + 1.03217 w_k1_MAT_s103
                                 + 1.37269 w_k1_MSFT_s103
                                 + 1.09881 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.820773 w_k1_ADM_s103
                                 + 0.956248 w_k1_AXP_s103
                                 + 0.810797 w_k1_DIS_s103
                                 + 0.910951 w_k1_FDX_s103
                                 + 1.04383 w_k1_IBM_s103
                                 + 0.985448 w_k1_INTC_s103
                                 + 0.881485 w_k1_KO_s103
                                 + 0.871145 w_k1_MAT_s103
                                 + 0.953012 w_k1_MSFT_s103
                                 + 0.95734 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.08587 w_k1_ADM_s103
                                 + 1.11745 w_k1_AXP_s103
                                 + 0.859782 w_k1_DIS_s103
                                 + 1.05735 w_k1_FDX_s103
                                 + 0.895225 w_k1_IBM_s103
                                 + 1.19181 w_k1_INTC_s103
                                 + 0.887153 w_k1_KO_s103
                                 + 0.645133 w_k1_MAT_s103
                                 + 0.934198 w_k1_MSFT_s103
                                 + 1.05671 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.03022 w_k1_ADM_s103
                                 + 0.961333 w_k1_AXP_s103
                                 + 1.16141 w_k1_DIS_s103
                                 + 1.05475 w_k1_FDX_s103
                                 + 0.977054 w_k1_IBM_s103
                                 + 0.977324 w_k1_INTC_s103
                                 + 1.01608 w_k1_KO_s103
                                 + 0.810395 w_k1_MAT_s103
                                 + 1.05654 w_k1_MSFT_s103
                                 + 1.09989 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.13476 w_k1_ADM_s103
                                 + 1.58956 w_k1_AXP_s103
                                 + 1.57334 w_k1_DIS_s103
                                 + 1.34041 w_k1_FDX_s103
                                 + 1.40453 w_k1_IBM_s103
                                 + 1.3139 w_k1_INTC_s103 + 1.4622 w_k1_KO_s103
                                 + 1.20942 w_k1_MAT_s103
                                 + 1.33045 w_k1_MSFT_s103
                                 + 1.32549 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.9611 w_k1_ADM_s103
                                 + 0.916854 w_k1_AXP_s103
                                 + 1.11017 w_k1_DIS_s103
                                 + 0.965514 w_k1_FDX_s103
                                 + 1.31965 w_k1_IBM_s103
                                 + 1.12216 w_k1_INTC_s103
                                 + 1.28758 w_k1_KO_s103 + 1.32452 w_k1_MAT_s103
                                 + 1.09102 w_k1_MSFT_s103
                                 + 1.18659 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.837469 w_k1_ADM_s104
                                 + 0.832029 w_k1_AXP_s104
                                 + 1.06066 w_k1_DIS_s104
                                 + 0.99881 w_k1_FDX_s104
                                 + 0.930017 w_k1_IBM_s104
                                 + 0.872177 w_k1_INTC_s104
                                 + 0.893698 w_k1_KO_s104
                                 + 0.979748 w_k1_MAT_s104
                                 + 1.02138 w_k1_MSFT_s104
                                 + 0.932773 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.727453 w_k1_ADM_s104
                                 + 0.992151 w_k1_AXP_s104
                                 + 0.745385 w_k1_DIS_s104
                                 + 0.809619 w_k1_FDX_s104
                                 + 0.906863 w_k1_IBM_s104
                                 + 0.836225 w_k1_INTC_s104
                                 + 0.791941 w_k1_KO_s104
                                 + 0.725029 w_k1_MAT_s104
                                 + 0.765444 w_k1_MSFT_s104
                                 + 0.945316 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.791348 w_k1_ADM_s104
                                 + 0.636947 w_k1_AXP_s104
                                 + 0.681796 w_k1_DIS_s104
                                 + 0.761603 w_k1_FDX_s104
                                 + 0.775269 w_k1_IBM_s104
                                 + 0.889977 w_k1_INTC_s104
                                 + 0.801008 w_k1_KO_s104
                                 + 0.626337 w_k1_MAT_s104
                                 + 0.851115 w_k1_MSFT_s104
                                 + 0.823285 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.843207 w_k1_ADM_s104
                                 + 1.13212 w_k1_AXP_s104
                                 + 1.34161 w_k1_DIS_s104
                                 + 1.18086 w_k1_FDX_s104
                                 + 1.70837 w_k1_IBM_s104
                                 + 1.26761 w_k1_INTC_s104
                                 + 1.25806 w_k1_KO_s104 + 1.05223 w_k1_MAT_s104
                                 + 1.09776 w_k1_MSFT_s104
                                 + 1.10879 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.782521 w_k1_ADM_s104
                                 + 1.02967 w_k1_AXP_s104
                                 + 0.911479 w_k1_DIS_s104
                                 + 0.920165 w_k1_FDX_s104
                                 + 1.2329 w_k1_IBM_s104
                                 + 1.01265 w_k1_INTC_s104
                                 + 1.10766 w_k1_KO_s104 + 1.21121 w_k1_MAT_s104
                                 + 0.899619 w_k1_MSFT_s104
                                 + 0.968427 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.944504 w_k1_ADM_s104
                                 + 0.747415 w_k1_AXP_s104
                                 + 0.898538 w_k1_DIS_s104
                                 + 0.993306 w_k1_FDX_s104
                                 + 0.617388 w_k1_IBM_s104
                                 + 1.02079 w_k1_INTC_s104
                                 + 0.746998 w_k1_KO_s104
                                 + 0.506222 w_k1_MAT_s104
                                 + 0.950521 w_k1_MSFT_s104
                                 + 0.916625 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.952131 w_k1_ADM_s104
                                 + 1.33576 w_k1_AXP_s104
                                 + 1.36105 w_k1_DIS_s104
                                 + 1.24484 w_k1_FDX_s104
                                 + 0.981906 w_k1_IBM_s104
                                 + 1.23598 w_k1_INTC_s104 + 1.1619 w_k1_KO_s104
                                 + 0.803662 w_k1_MAT_s104
                                 + 1.17495 w_k1_MSFT_s104 + 1.0893 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.924501 w_k1_ADM_s104
                                 + 1.17553 w_k1_AXP_s104
                                 + 0.959199 w_k1_DIS_s104
                                 + 1.02407 w_k1_FDX_s104
                                 + 0.775895 w_k1_IBM_s104
                                 + 0.950798 w_k1_INTC_s104
                                 + 0.864033 w_k1_KO_s104
                                 + 0.799109 w_k1_MAT_s104
                                 + 0.880423 w_k1_MSFT_s104
                                 + 1.03253 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.999616 w_k1_ADM_s105
                                 + 0.909153 w_k1_AXP_s105
                                 + 1.06541 w_k1_DIS_s105
                                 + 1.01318 w_k1_FDX_s105
                                 + 1.45046 w_k1_IBM_s105
                                 + 1.01817 w_k1_INTC_s105
                                 + 1.14095 w_k1_KO_s105 + 1.13033 w_k1_MAT_s105
                                 + 1.03457 w_k1_MSFT_s105
                                 + 1.08405 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.873438 w_k1_ADM_s105
                                 + 0.658537 w_k1_AXP_s105
                                 + 0.844373 w_k1_DIS_s105
                                 + 0.880259 w_k1_FDX_s105
                                 + 0.725088 w_k1_IBM_s105
                                 + 0.705475 w_k1_INTC_s105
                                 + 0.86919 w_k1_KO_s105 + 1.08199 w_k1_MAT_s105
                                 + 0.915665 w_k1_MSFT_s105
                                 + 0.959477 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.05618 w_k1_ADM_s105
                                 + 1.15132 w_k1_AXP_s105
                                 + 0.861581 w_k1_DIS_s105
                                 + 1.05516 w_k1_FDX_s105
                                 + 0.811122 w_k1_IBM_s105
                                 + 0.92401 w_k1_INTC_s105
                                 + 0.84243 w_k1_KO_s105 + 1.05614 w_k1_MAT_s105
                                 + 0.970982 w_k1_MSFT_s105
                                 + 1.02446 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.23606 w_k1_ADM_s105
                                 + 1.38236 w_k1_AXP_s105
                                 + 1.24297 w_k1_DIS_s105 + 1.2301 w_k1_FDX_s105
                                 + 1.06746 w_k1_IBM_s105
                                 + 1.18805 w_k1_INTC_s105
                                 + 1.03986 w_k1_KO_s105 + 1.41238 w_k1_MAT_s105
                                 + 1.07817 w_k1_MSFT_s105 + 1.1164 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.828175 w_k1_ADM_s105
                                 + 0.899046 w_k1_AXP_s105
                                 + 0.763144 w_k1_DIS_s105
                                 + 0.914453 w_k1_FDX_s105
                                 + 0.983436 w_k1_IBM_s105
                                 + 0.982954 w_k1_INTC_s105
                                 + 0.786831 w_k1_KO_s105
                                 + 1.35129 w_k1_MAT_s105
                                 + 1.00594 w_k1_MSFT_s105
                                 + 0.946213 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.41785 w_k1_ADM_s105
                                 + 1.87391 w_k1_AXP_s105
                                 + 1.78162 w_k1_DIS_s105
                                 + 1.46116 w_k1_FDX_s105
                                 + 1.52922 w_k1_IBM_s105
                                 + 1.37739 w_k1_INTC_s105
                                 + 1.45036 w_k1_KO_s105 + 2.17563 w_k1_MAT_s105
                                 + 1.45282 w_k1_MSFT_s105
                                 + 1.60238 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.02014 w_k1_ADM_s105
                                 + 1.10503 w_k1_AXP_s105
                                 + 1.51559 w_k1_DIS_s105
                                 + 1.25005 w_k1_FDX_s105
                                 + 1.34825 w_k1_IBM_s105
                                 + 1.13285 w_k1_INTC_s105 + 1.167 w_k1_KO_s105
                                 + 1.36067 w_k1_MAT_s105
                                 + 1.32064 w_k1_MSFT_s105
                                 + 1.06483 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.785659 w_k1_ADM_s105
                                 + 0.6603 w_k1_AXP_s105
                                 + 0.538526 w_k1_DIS_s105
                                 + 0.692724 w_k1_FDX_s105
                                 + 0.843371 w_k1_IBM_s105
                                 + 0.795064 w_k1_INTC_s105
                                 + 0.652798 w_k1_KO_s105 + 0.7838 w_k1_MAT_s105
                                 + 0.698797 w_k1_MSFT_s105
                                 + 0.832372 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.3047 w_k1_ADM_s106
                                 + 1.12127 w_k1_AXP_s106
                                 + 1.28603 w_k1_DIS_s106
                                 + 1.34706 w_k1_FDX_s106
                                 + 0.780756 w_k1_IBM_s106
                                 + 1.18327 w_k1_INTC_s106
                                 + 1.01514 w_k1_KO_s106
                                 + 0.744928 w_k1_MAT_s106
                                 + 1.22669 w_k1_MSFT_s106
                                 + 1.06147 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.833141 w_k1_ADM_s106
                                 + 0.725706 w_k1_AXP_s106
                                 + 0.61165 w_k1_DIS_s106
                                 + 0.735544 w_k1_FDX_s106
                                 + 0.75363 w_k1_IBM_s106
                                 + 0.701164 w_k1_INTC_s106
                                 + 0.688925 w_k1_KO_s106
                                 + 0.57752 w_k1_MAT_s106
                                 + 0.686886 w_k1_MSFT_s106
                                 + 0.728478 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.24104 w_k1_ADM_s106
                                 + 1.49665 w_k1_AXP_s106
                                 + 1.63621 w_k1_DIS_s106
                                 + 1.28188 w_k1_FDX_s106
                                 + 1.53249 w_k1_IBM_s106
                                 + 1.24787 w_k1_INTC_s106
                                 + 1.76396 w_k1_KO_s106 + 1.44564 w_k1_MAT_s106
                                 + 1.30109 w_k1_MSFT_s106
                                 + 1.30762 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.906986 w_k1_ADM_s106
                                 + 1.14945 w_k1_AXP_s106
                                 + 1.23984 w_k1_DIS_s106
                                 + 1.02615 w_k1_FDX_s106
                                 + 1.51342 w_k1_IBM_s106
                                 + 0.813521 w_k1_INTC_s106
                                 + 1.17197 w_k1_KO_s106 + 1.30112 w_k1_MAT_s106
                                 + 0.916062 w_k1_MSFT_s106
                                 + 0.997376 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.964387 w_k1_ADM_s106
                                 + 0.986607 w_k1_AXP_s106
                                 + 0.970982 w_k1_DIS_s106
                                 + 1.02519 w_k1_FDX_s106
                                 + 0.986909 w_k1_IBM_s106
                                 + 0.854655 w_k1_INTC_s106
                                 + 0.907644 w_k1_KO_s106
                                 + 0.798865 w_k1_MAT_s106
                                 + 0.843836 w_k1_MSFT_s106
                                 + 0.933692 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.964411 w_k1_ADM_s106
                                 + 0.864011 w_k1_AXP_s106
                                 + 0.762502 w_k1_DIS_s106
                                 + 0.826635 w_k1_FDX_s106
                                 + 1.33955 w_k1_IBM_s106
                                 + 0.838365 w_k1_INTC_s106
                                 + 1.15458 w_k1_KO_s106 + 1.01392 w_k1_MAT_s106
                                 + 0.840084 w_k1_MSFT_s106
                                 + 0.960232 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.981519 w_k1_ADM_s106
                                 + 1.09738 w_k1_AXP_s106
                                 + 0.679396 w_k1_DIS_s106
                                 + 0.915895 w_k1_FDX_s106
                                 + 0.983344 w_k1_IBM_s106
                                 + 0.847298 w_k1_INTC_s106
                                 + 0.778751 w_k1_KO_s106
                                 + 0.857568 w_k1_MAT_s106
                                 + 0.655818 w_k1_MSFT_s106
                                 + 0.829924 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.971904 w_k1_ADM_s106
                                 + 0.769724 w_k1_AXP_s106
                                 + 0.858777 w_k1_DIS_s106
                                 + 0.942332 w_k1_FDX_s106
                                 + 0.766047 w_k1_IBM_s106
                                 + 0.7675 w_k1_INTC_s106
                                 + 0.993722 w_k1_KO_s106
                                 + 0.64275 w_k1_MAT_s106
                                 + 0.908243 w_k1_MSFT_s106
                                 + 0.915756 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.47364 w_k1_ADM_s107
                                 + 1.62461 w_k1_AXP_s107
                                 + 1.33935 w_k1_DIS_s107
                                 + 1.27214 w_k1_FDX_s107
                                 + 1.10719 w_k1_IBM_s107
                                 + 1.02031 w_k1_INTC_s107 + 1.051 w_k1_KO_s107
                                 + 1.47418 w_k1_MAT_s107
                                 + 1.1766 w_k1_MSFT_s107 + 1.18876 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.785499 w_k1_ADM_s107
                                 + 0.683098 w_k1_AXP_s107
                                 + 0.597566 w_k1_DIS_s107
                                 + 0.78444 w_k1_FDX_s107
                                 + 0.688487 w_k1_IBM_s107
                                 + 0.752648 w_k1_INTC_s107
                                 + 0.694696 w_k1_KO_s107
                                 + 0.617628 w_k1_MAT_s107
                                 + 0.72848 w_k1_MSFT_s107
                                 + 0.837956 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.953075 w_k1_ADM_s107
                                 + 1.12864 w_k1_AXP_s107
                                 + 1.07028 w_k1_DIS_s107
                                 + 1.05378 w_k1_FDX_s107
                                 + 0.983758 w_k1_IBM_s107
                                 + 0.84439 w_k1_INTC_s107
                                 + 0.913513 w_k1_KO_s107 + 1.9113 w_k1_MAT_s107
                                 + 1.02768 w_k1_MSFT_s107
                                 + 1.07012 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.997198 w_k1_ADM_s107
                                 + 1.22873 w_k1_AXP_s107
                                 + 1.18144 w_k1_DIS_s107
                                 + 1.02187 w_k1_FDX_s107
                                 + 1.43579 w_k1_IBM_s107
                                 + 1.0547 w_k1_INTC_s107 + 1.38088 w_k1_KO_s107
                                 + 2.36248 w_k1_MAT_s107
                                 + 1.17891 w_k1_MSFT_s107
                                 + 1.03425 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.779603 w_k1_ADM_s107
                                 + 0.62033 w_k1_AXP_s107
                                 + 0.61591 w_k1_DIS_s107
                                 + 0.709008 w_k1_FDX_s107
                                 + 0.966373 w_k1_IBM_s107
                                 + 0.829055 w_k1_INTC_s107
                                 + 0.799752 w_k1_KO_s107
                                 + 1.01149 w_k1_MAT_s107
                                 + 0.780966 w_k1_MSFT_s107
                                 + 0.799398 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.945676 w_k1_ADM_s107
                                 + 0.969707 w_k1_AXP_s107
                                 + 0.914759 w_k1_DIS_s107
                                 + 0.984717 w_k1_FDX_s107
                                 + 0.787843 w_k1_IBM_s107
                                 + 1.14383 w_k1_INTC_s107
                                 + 0.883455 w_k1_KO_s107
                                 + 1.03689 w_k1_MAT_s107
                                 + 1.01243 w_k1_MSFT_s107
                                 + 1.01729 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.888276 w_k1_ADM_s107
                                 + 0.710606 w_k1_AXP_s107
                                 + 0.902897 w_k1_DIS_s107
                                 + 0.795896 w_k1_FDX_s107
                                 + 1.20762 w_k1_IBM_s107
                                 + 0.795938 w_k1_INTC_s107
                                 + 0.948594 w_k1_KO_s107 + 1.4049 w_k1_MAT_s107
                                 + 0.950948 w_k1_MSFT_s107
                                 + 0.941359 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.17068 w_k1_ADM_s107
                                 + 1.05371 w_k1_AXP_s107
                                 + 1.10087 w_k1_DIS_s107
                                 + 1.10463 w_k1_FDX_s107
                                 + 0.984803 w_k1_IBM_s107
                                 + 1.15338 w_k1_INTC_s107
                                 + 1.02136 w_k1_KO_s107
                                 + 0.984516 w_k1_MAT_s107
                                 + 1.23596 w_k1_MSFT_s107
                                 + 0.989006 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.886923 w_k1_ADM_s110
                                 + 0.963727 w_k1_AXP_s110
                                 + 0.821892 w_k1_DIS_s110
                                 + 0.819196 w_k1_FDX_s110
                                 + 1.09257 w_k1_IBM_s110
                                 + 0.788764 w_k1_INTC_s110
                                 + 0.828711 w_k1_KO_s110
                                 + 0.880016 w_k1_MAT_s110
                                 + 0.740075 w_k1_MSFT_s110
                                 + 0.807973 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.965024 w_k1_ADM_s110
                                 + 0.849852 w_k1_AXP_s110
                                 + 0.747032 w_k1_DIS_s110
                                 + 0.824212 w_k1_FDX_s110
                                 + 0.604036 w_k1_IBM_s110
                                 + 0.782855 w_k1_INTC_s110
                                 + 0.722569 w_k1_KO_s110
                                 + 0.57118 w_k1_MAT_s110
                                 + 0.714819 w_k1_MSFT_s110
                                 + 0.803964 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.666154 w_k1_ADM_s110
                                 + 0.673041 w_k1_AXP_s110
                                 + 0.695809 w_k1_DIS_s110
                                 + 0.771749 w_k1_FDX_s110
                                 + 0.540318 w_k1_IBM_s110
                                 + 0.838641 w_k1_INTC_s110
                                 + 0.639651 w_k1_KO_s110
                                 + 0.63659 w_k1_MAT_s110
                                 + 0.798409 w_k1_MSFT_s110
                                 + 0.674876 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.17892 w_k1_ADM_s110
                                 + 1.09729 w_k1_AXP_s110
                                 + 1.31303 w_k1_DIS_s110
                                 + 1.09219 w_k1_FDX_s110
                                 + 1.07705 w_k1_IBM_s110 + 1.127 w_k1_INTC_s110
                                 + 1.15285 w_k1_KO_s110 + 1.28974 w_k1_MAT_s110
                                 + 1.05721 w_k1_MSFT_s110
                                 + 1.11877 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.18755 w_k1_ADM_s110
                                 + 1.44521 w_k1_AXP_s110
                                 + 1.35668 w_k1_DIS_s110
                                 + 1.15518 w_k1_FDX_s110
                                 + 1.41183 w_k1_IBM_s110
                                 + 1.18457 w_k1_INTC_s110
                                 + 1.20246 w_k1_KO_s110 + 1.11063 w_k1_MAT_s110
                                 + 0.912599 w_k1_MSFT_s110
                                 + 1.04624 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.21137 w_k1_ADM_s110
                                 + 1.33783 w_k1_AXP_s110
                                 + 1.13678 w_k1_DIS_s110
                                 + 0.963798 w_k1_FDX_s110
                                 + 1.7251 w_k1_IBM_s110
                                 + 0.938283 w_k1_INTC_s110
                                 + 1.3545 w_k1_KO_s110 + 1.86154 w_k1_MAT_s110
                                 + 0.966805 w_k1_MSFT_s110
                                 + 1.04001 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.04367 w_k1_ADM_s110
                                 + 0.860037 w_k1_AXP_s110
                                 + 0.858579 w_k1_DIS_s110
                                 + 0.890773 w_k1_FDX_s110
                                 + 0.71854 w_k1_IBM_s110
                                 + 0.95916 w_k1_INTC_s110
                                 + 0.832503 w_k1_KO_s110
                                 + 0.813684 w_k1_MAT_s110
                                 + 0.985642 w_k1_MSFT_s110
                                 + 0.952046 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.01425 w_k1_ADM_s110
                                 + 0.966854 w_k1_AXP_s110
                                 + 1.02384 w_k1_DIS_s110
                                 + 0.974974 w_k1_FDX_s110
                                 + 0.83842 w_k1_IBM_s110
                                 + 0.970511 w_k1_INTC_s110
                                 + 0.868837 w_k1_KO_s110
                                 + 0.907473 w_k1_MAT_s110
                                 + 1.00517 w_k1_MSFT_s110
                                 + 0.956071 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.72045 w_k1_ADM_s111
                                 + 0.839372 w_k1_AXP_s111
                                 + 0.660669 w_k1_DIS_s111
                                 + 0.887292 w_k1_FDX_s111
                                 + 0.778778 w_k1_IBM_s111
                                 + 1.09732 w_k1_INTC_s111
                                 + 0.839243 w_k1_KO_s111
                                 + 0.66455 w_k1_MAT_s111
                                 + 0.902129 w_k1_MSFT_s111
                                 + 0.904455 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.911939 w_k1_ADM_s111
                                 + 1.0366 w_k1_AXP_s111
                                 + 0.952688 w_k1_DIS_s111
                                 + 1.16355 w_k1_FDX_s111 + 0.771 w_k1_IBM_s111
                                 + 0.915781 w_k1_INTC_s111
                                 + 0.839296 w_k1_KO_s111
                                 + 0.853343 w_k1_MAT_s111
                                 + 0.903225 w_k1_MSFT_s111
                                 + 1.00757 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.916949 w_k1_ADM_s111
                                 + 1.55053 w_k1_AXP_s111
                                 + 1.02149 w_k1_DIS_s111
                                 + 1.21403 w_k1_FDX_s111
                                 + 1.31259 w_k1_IBM_s111
                                 + 0.905221 w_k1_INTC_s111
                                 + 0.99776 w_k1_KO_s111 + 1.7556 w_k1_MAT_s111
                                 + 1.00705 w_k1_MSFT_s111
                                 + 1.11054 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.608595 w_k1_ADM_s111
                                 + 0.727169 w_k1_AXP_s111
                                 + 0.57999 w_k1_DIS_s111
                                 + 0.761072 w_k1_FDX_s111
                                 + 0.843994 w_k1_IBM_s111
                                 + 0.913488 w_k1_INTC_s111
                                 + 0.722586 w_k1_KO_s111
                                 + 0.692038 w_k1_MAT_s111
                                 + 0.817627 w_k1_MSFT_s111
                                 + 0.792693 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.951129 w_k1_ADM_s111
                                 + 0.844412 w_k1_AXP_s111
                                 + 0.925705 w_k1_DIS_s111
                                 + 0.972347 w_k1_FDX_s111
                                 + 1.00566 w_k1_IBM_s111
                                 + 1.07838 w_k1_INTC_s111
                                 + 1.00676 w_k1_KO_s111
                                 + 0.598947 w_k1_MAT_s111
                                 + 1.03204 w_k1_MSFT_s111
                                 + 1.00072 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.837092 w_k1_ADM_s111
                                 + 1.11818 w_k1_AXP_s111
                                 + 1.32106 w_k1_DIS_s111
                                 + 1.23877 w_k1_FDX_s111
                                 + 0.955493 w_k1_IBM_s111
                                 + 1.13519 w_k1_INTC_s111
                                 + 1.00155 w_k1_KO_s111 + 1.03362 w_k1_MAT_s111
                                 + 1.18664 w_k1_MSFT_s111
                                 + 1.09926 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.874014 w_k1_ADM_s111
                                 + 1.16562 w_k1_AXP_s111
                                 + 1.58521 w_k1_DIS_s111
                                 + 1.23462 w_k1_FDX_s111 + 1.4484 w_k1_IBM_s111
                                 + 1.10125 w_k1_INTC_s111
                                 + 1.36714 w_k1_KO_s111 + 1.61963 w_k1_MAT_s111
                                 + 1.33221 w_k1_MSFT_s111
                                 + 1.30465 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.745985 w_k1_ADM_s111
                                 + 0.920296 w_k1_AXP_s111
                                 + 0.848972 w_k1_DIS_s111
                                 + 0.978407 w_k1_FDX_s111
                                 + 1.16753 w_k1_IBM_s111
                                 + 0.988968 w_k1_INTC_s111
                                 + 1.06235 w_k1_KO_s111 + 1.06529 w_k1_MAT_s111
                                 + 0.976629 w_k1_MSFT_s111
                                 + 1.07404 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.851571 w_k1_ADM_s112
                                 + 0.830703 w_k1_AXP_s112
                                 + 0.862422 w_k1_DIS_s112
                                 + 0.840894 w_k1_FDX_s112
                                 + 1.00789 w_k1_IBM_s112
                                 + 0.970109 w_k1_INTC_s112
                                 + 0.943007 w_k1_KO_s112
                                 + 1.30931 w_k1_MAT_s112
                                 + 0.895136 w_k1_MSFT_s112
                                 + 1.08047 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.836878 w_k1_ADM_s112
                                 + 0.946907 w_k1_AXP_s112
                                 + 0.892638 w_k1_DIS_s112
                                 + 0.913774 w_k1_FDX_s112
                                 + 0.761692 w_k1_IBM_s112
                                 + 0.892539 w_k1_INTC_s112
                                 + 0.714319 w_k1_KO_s112
                                 + 0.724031 w_k1_MAT_s112
                                 + 0.853668 w_k1_MSFT_s112
                                 + 1.02634 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.06977 w_k1_ADM_s112
                                 + 1.25212 w_k1_AXP_s112
                                 + 1.35009 w_k1_DIS_s112
                                 + 1.17231 w_k1_FDX_s112
                                 + 1.17238 w_k1_IBM_s112
                                 + 0.984079 w_k1_INTC_s112
                                 + 1.11168 w_k1_KO_s112 + 1.4704 w_k1_MAT_s112
                                 + 1.03408 w_k1_MSFT_s112
                                 + 1.16276 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.884061 w_k1_ADM_s112
                                 + 0.869073 w_k1_AXP_s112
                                 + 0.703851 w_k1_DIS_s112
                                 + 0.861882 w_k1_FDX_s112
                                 + 1.01924 w_k1_IBM_s112
                                 + 0.884035 w_k1_INTC_s112
                                 + 0.877662 w_k1_KO_s112
                                 + 1.30254 w_k1_MAT_s112
                                 + 0.707226 w_k1_MSFT_s112
                                 + 1.00015 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.09898 w_k1_ADM_s112
                                 + 2.11652 w_k1_AXP_s112
                                 + 1.97635 w_k1_DIS_s112 + 1.7071 w_k1_FDX_s112
                                 + 1.67446 w_k1_IBM_s112
                                 + 1.70544 w_k1_INTC_s112 + 1.56 w_k1_KO_s112
                                 + 2.40454 w_k1_MAT_s112
                                 + 1.41736 w_k1_MSFT_s112 + 1.5278 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.860314 w_k1_ADM_s112
                                 + 1.15495 w_k1_AXP_s112
                                 + 0.891727 w_k1_DIS_s112
                                 + 0.918968 w_k1_FDX_s112
                                 + 1.51953 w_k1_IBM_s112
                                 + 0.857102 w_k1_INTC_s112
                                 + 1.01149 w_k1_KO_s112 + 1.57127 w_k1_MAT_s112
                                 + 0.67146 w_k1_MSFT_s112
                                 + 1.04875 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.726517 w_k1_ADM_s112
                                 + 0.955956 w_k1_AXP_s112
                                 + 0.721433 w_k1_DIS_s112
                                 + 0.986927 w_k1_FDX_s112
                                 + 0.556073 w_k1_IBM_s112
                                 + 0.855742 w_k1_INTC_s112
                                 + 0.579009 w_k1_KO_s112
                                 + 0.560195 w_k1_MAT_s112
                                 + 0.72352 w_k1_MSFT_s112
                                 + 0.813646 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.29989 w_k1_ADM_s112
                                 + 1.2769 w_k1_AXP_s112 + 1.2782 w_k1_DIS_s112
                                 + 1.43229 w_k1_FDX_s112
                                 + 0.746565 w_k1_IBM_s112
                                 + 1.173 w_k1_INTC_s112 + 0.93453 w_k1_KO_s112
                                 + 0.896782 w_k1_MAT_s112
                                 + 1.10068 w_k1_MSFT_s112
                                 + 1.02335 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02391 w_k1_ADM_s113
                                 + 0.783408 w_k1_AXP_s113
                                 + 0.721321 w_k1_DIS_s113
                                 + 0.869168 w_k1_FDX_s113
                                 + 0.731858 w_k1_IBM_s113
                                 + 0.940149 w_k1_INTC_s113
                                 + 0.712613 w_k1_KO_s113
                                 + 0.624341 w_k1_MAT_s113
                                 + 0.912901 w_k1_MSFT_s113
                                 + 0.82824 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.18091 w_k1_ADM_s113
                                 + 1.69972 w_k1_AXP_s113
                                 + 1.57268 w_k1_DIS_s113
                                 + 1.36296 w_k1_FDX_s113
                                 + 1.32622 w_k1_IBM_s113
                                 + 1.77894 w_k1_INTC_s113
                                 + 1.44644 w_k1_KO_s113 + 1.4517 w_k1_MAT_s113
                                 + 1.36876 w_k1_MSFT_s113
                                 + 1.21096 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.82632 w_k1_ADM_s113
                                 + 1.14649 w_k1_AXP_s113
                                 + 1.17472 w_k1_DIS_s113
                                 + 1.02572 w_k1_FDX_s113
                                 + 1.83148 w_k1_IBM_s113
                                 + 1.2622 w_k1_INTC_s113 + 1.3621 w_k1_KO_s113
                                 + 1.67853 w_k1_MAT_s113
                                 + 1.0338 w_k1_MSFT_s113 + 1.12049 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.998682 w_k1_ADM_s113
                                 + 0.96171 w_k1_AXP_s113
                                 + 0.76454 w_k1_DIS_s113
                                 + 0.837286 w_k1_FDX_s113
                                 + 1.11958 w_k1_IBM_s113
                                 + 0.981743 w_k1_INTC_s113
                                 + 0.764088 w_k1_KO_s113
                                 + 1.04756 w_k1_MAT_s113
                                 + 0.905197 w_k1_MSFT_s113
                                 + 0.81803 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.17624 w_k1_ADM_s113
                                 + 1.29415 w_k1_AXP_s113
                                 + 1.46347 w_k1_DIS_s113
                                 + 1.17197 w_k1_FDX_s113
                                 + 1.21091 w_k1_IBM_s113
                                 + 1.06296 w_k1_INTC_s113
                                 + 1.12645 w_k1_KO_s113 + 1.20928 w_k1_MAT_s113
                                 + 1.19641 w_k1_MSFT_s113
                                 + 0.977978 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.15039 w_k1_ADM_s113
                                 + 1.32396 w_k1_AXP_s113 + 1.1678 w_k1_DIS_s113
                                 + 1.23011 w_k1_FDX_s113
                                 + 0.810348 w_k1_IBM_s113
                                 + 1.11545 w_k1_INTC_s113
                                 + 0.915349 w_k1_KO_s113
                                 + 0.755487 w_k1_MAT_s113
                                 + 1.08937 w_k1_MSFT_s113
                                 + 0.961463 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.16601 w_k1_ADM_s113
                                 + 1.19506 w_k1_AXP_s113
                                 + 1.00011 w_k1_DIS_s113
                                 + 0.973414 w_k1_FDX_s113
                                 + 1.22224 w_k1_IBM_s113
                                 + 1.05299 w_k1_INTC_s113
                                 + 1.04738 w_k1_KO_s113 + 1.32296 w_k1_MAT_s113
                                 + 1.01102 w_k1_MSFT_s113
                                 + 0.980407 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.37034 w_k1_ADM_s113
                                 + 1.0034 w_k1_AXP_s113 + 1.15938 w_k1_DIS_s113
                                 + 1.14625 w_k1_FDX_s113
                                 + 0.985846 w_k1_IBM_s113
                                 + 1.09341 w_k1_INTC_s113
                                 + 0.963567 w_k1_KO_s113
                                 + 0.797268 w_k1_MAT_s113
                                 + 1.14457 w_k1_MSFT_s113
                                 + 1.10246 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.31804 w_k1_ADM_s114
                                 + 1.38226 w_k1_AXP_s114
                                 + 1.68347 w_k1_DIS_s114
                                 + 1.27072 w_k1_FDX_s114
                                 + 1.35761 w_k1_IBM_s114
                                 + 1.24296 w_k1_INTC_s114
                                 + 1.48854 w_k1_KO_s114 + 1.50626 w_k1_MAT_s114
                                 + 1.4061 w_k1_MSFT_s114 + 1.18035 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.1244 w_k1_ADM_s114
                                 + 1.10235 w_k1_AXP_s114 + 1.0235 w_k1_DIS_s114
                                 + 1.04363 w_k1_FDX_s114
                                 + 1.42566 w_k1_IBM_s114
                                 + 1.07833 w_k1_INTC_s114
                                 + 1.26236 w_k1_KO_s114 + 1.10166 w_k1_MAT_s114
                                 + 1.09945 w_k1_MSFT_s114
                                 + 0.996259 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.43257 w_k1_ADM_s114
                                 + 1.3023 w_k1_AXP_s114 + 1.08733 w_k1_DIS_s114
                                 + 1.14551 w_k1_FDX_s114
                                 + 0.979343 w_k1_IBM_s114
                                 + 0.932512 w_k1_INTC_s114
                                 + 1.0824 w_k1_KO_s114 + 0.822089 w_k1_MAT_s114
                                 + 0.972171 w_k1_MSFT_s114
                                 + 0.941569 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.06946 w_k1_ADM_s114
                                 + 1.17184 w_k1_AXP_s114
                                 + 0.72874 w_k1_DIS_s114
                                 + 0.920116 w_k1_FDX_s114
                                 + 1.1181 w_k1_IBM_s114
                                 + 0.767356 w_k1_INTC_s114
                                 + 0.80661 w_k1_KO_s114 + 1.15396 w_k1_MAT_s114
                                 + 0.723753 w_k1_MSFT_s114
                                 + 0.93869 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.53882 w_k1_ADM_s114
                                 + 1.0131 w_k1_AXP_s114 + 1.51006 w_k1_DIS_s114
                                 + 1.37807 w_k1_FDX_s114
                                 + 0.868373 w_k1_IBM_s114
                                 + 1.36039 w_k1_INTC_s114
                                 + 1.22144 w_k1_KO_s114
                                 + 0.506559 w_k1_MAT_s114
                                 + 1.315 w_k1_MSFT_s114 + 1.06209 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.882387 w_k1_ADM_s114
                                 + 0.788096 w_k1_AXP_s114
                                 + 0.763161 w_k1_DIS_s114
                                 + 0.854673 w_k1_FDX_s114
                                 + 1.01354 w_k1_IBM_s114
                                 + 1.15508 w_k1_INTC_s114
                                 + 0.986545 w_k1_KO_s114
                                 + 0.875111 w_k1_MAT_s114
                                 + 0.937401 w_k1_MSFT_s114
                                 + 0.937752 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.06098 w_k1_ADM_s114
                                 + 0.817003 w_k1_AXP_s114
                                 + 0.642714 w_k1_DIS_s114
                                 + 0.804232 w_k1_FDX_s114
                                 + 0.724153 w_k1_IBM_s114
                                 + 0.817355 w_k1_INTC_s114
                                 + 0.847323 w_k1_KO_s114
                                 + 0.727889 w_k1_MAT_s114
                                 + 0.836703 w_k1_MSFT_s114
                                 + 0.81399 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.28081 w_k1_ADM_s114
                                 + 1.05903 w_k1_AXP_s114
                                 + 0.913011 w_k1_DIS_s114
                                 + 1.09262 w_k1_FDX_s114
                                 + 0.742798 w_k1_IBM_s114
                                 + 0.975677 w_k1_INTC_s114
                                 + 0.927113 w_k1_KO_s114
                                 + 0.786739 w_k1_MAT_s114
                                 + 1.00942 w_k1_MSFT_s114
                                 + 0.887555 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.34247 w_k1_ADM_s115
                                 + 1.43014 w_k1_AXP_s115
                                 + 1.80127 w_k1_DIS_s115
                                 + 1.39937 w_k1_FDX_s115
                                 + 1.29548 w_k1_IBM_s115
                                 + 1.18244 w_k1_INTC_s115
                                 + 1.37194 w_k1_KO_s115 + 1.29981 w_k1_MAT_s115
                                 + 1.20614 w_k1_MSFT_s115 + 1.2238 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.734295 w_k1_ADM_s115
                                 + 0.649028 w_k1_AXP_s115
                                 + 0.652781 w_k1_DIS_s115
                                 + 0.756315 w_k1_FDX_s115
                                 + 0.810678 w_k1_IBM_s115
                                 + 0.760623 w_k1_INTC_s115
                                 + 0.874406 w_k1_KO_s115
                                 + 0.822429 w_k1_MAT_s115
                                 + 0.833903 w_k1_MSFT_s115
                                 + 0.963954 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.18911 w_k1_ADM_s115
                                 + 1.0364 w_k1_AXP_s115
                                 + 0.887077 w_k1_DIS_s115
                                 + 1.12853 w_k1_FDX_s115
                                 + 1.03049 w_k1_IBM_s115
                                 + 1.08241 w_k1_INTC_s115 + 1.0385 w_k1_KO_s115
                                 + 1.21303 w_k1_MAT_s115
                                 + 0.909849 w_k1_MSFT_s115
                                 + 1.02825 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01706 w_k1_ADM_s115
                                 + 0.916495 w_k1_AXP_s115
                                 + 1.25382 w_k1_DIS_s115
                                 + 1.10269 w_k1_FDX_s115 + 1.0423 w_k1_IBM_s115
                                 + 0.97586 w_k1_INTC_s115
                                 + 1.24552 w_k1_KO_s115
                                 + 0.946843 w_k1_MAT_s115
                                 + 1.21389 w_k1_MSFT_s115
                                 + 1.15399 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.908352 w_k1_ADM_s115
                                 + 0.810743 w_k1_AXP_s115
                                 + 0.697028 w_k1_DIS_s115
                                 + 0.820705 w_k1_FDX_s115
                                 + 1.00977 w_k1_IBM_s115
                                 + 0.859469 w_k1_INTC_s115
                                 + 0.917774 w_k1_KO_s115
                                 + 0.984831 w_k1_MAT_s115
                                 + 0.724432 w_k1_MSFT_s115
                                 + 0.928473 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0237 w_k1_ADM_s115
                                 + 0.794069 w_k1_AXP_s115
                                 + 1.09011 w_k1_DIS_s115
                                 + 1.05472 w_k1_FDX_s115
                                 + 0.750182 w_k1_IBM_s115
                                 + 1.05678 w_k1_INTC_s115
                                 + 1.03923 w_k1_KO_s115
                                 + 0.731416 w_k1_MAT_s115
                                 + 1.00059 w_k1_MSFT_s115
                                 + 1.01605 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.969273 w_k1_ADM_s115
                                 + 0.977981 w_k1_AXP_s115
                                 + 0.719303 w_k1_DIS_s115
                                 + 1.03634 w_k1_FDX_s115
                                 + 0.643178 w_k1_IBM_s115
                                 + 0.799317 w_k1_INTC_s115
                                 + 0.680132 w_k1_KO_s115
                                 + 0.814302 w_k1_MAT_s115
                                 + 0.764128 w_k1_MSFT_s115
                                 + 0.9657 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.757227 w_k1_ADM_s115
                                 + 1.27581 w_k1_AXP_s115
                                 + 1.16438 w_k1_DIS_s115 + 1.0625 w_k1_FDX_s115
                                 + 1.4538 w_k1_IBM_s115
                                 + 0.962085 w_k1_INTC_s115
                                 + 1.17896 w_k1_KO_s115 + 2.08055 w_k1_MAT_s115
                                 + 0.898413 w_k1_MSFT_s115
                                 + 1.13822 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.03814 w_k1_ADM_s116
                                 + 0.708139 w_k1_AXP_s116
                                 + 0.679187 w_k1_DIS_s116
                                 + 0.841681 w_k1_FDX_s116
                                 + 0.714397 w_k1_IBM_s116
                                 + 0.962649 w_k1_INTC_s116
                                 + 0.861353 w_k1_KO_s116
                                 + 0.562042 w_k1_MAT_s116
                                 + 0.968796 w_k1_MSFT_s116
                                 + 0.898299 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.20324 w_k1_ADM_s116
                                 + 1.54819 w_k1_AXP_s116
                                 + 1.39466 w_k1_DIS_s116
                                 + 1.28405 w_k1_FDX_s116
                                 + 0.923326 w_k1_IBM_s116
                                 + 1.34372 w_k1_INTC_s116
                                 + 1.13923 w_k1_KO_s116 + 1.01223 w_k1_MAT_s116
                                 + 1.30015 w_k1_MSFT_s116
                                 + 1.32713 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.857603 w_k1_ADM_s116
                                 + 1.03718 w_k1_AXP_s116
                                 + 0.999811 w_k1_DIS_s116
                                 + 0.904764 w_k1_FDX_s116
                                 + 1.59365 w_k1_IBM_s116
                                 + 1.21575 w_k1_INTC_s116
                                 + 1.09139 w_k1_KO_s116 + 1.28915 w_k1_MAT_s116
                                 + 1.10339 w_k1_MSFT_s116
                                 + 1.01398 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.63917 w_k1_ADM_s116
                                 + 1.42575 w_k1_AXP_s116
                                 + 0.732585 w_k1_DIS_s116
                                 + 1.07361 w_k1_FDX_s116
                                 + 0.812162 w_k1_IBM_s116
                                 + 1.12631 w_k1_INTC_s116
                                 + 0.818788 w_k1_KO_s116
                                 + 1.16675 w_k1_MAT_s116
                                 + 1.00019 w_k1_MSFT_s116
                                 + 1.04141 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.29097 w_k1_ADM_s116
                                 + 1.1978 w_k1_AXP_s116 + 1.19069 w_k1_DIS_s116
                                 + 1.17006 w_k1_FDX_s116
                                 + 1.24094 w_k1_IBM_s116
                                 + 1.33108 w_k1_INTC_s116
                                 + 1.15773 w_k1_KO_s116
                                 + 0.999069 w_k1_MAT_s116
                                 + 1.08267 w_k1_MSFT_s116
                                 + 1.18359 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.797368 w_k1_ADM_s116
                                 + 0.754697 w_k1_AXP_s116
                                 + 0.59142 w_k1_DIS_s116
                                 + 0.664011 w_k1_FDX_s116
                                 + 0.952522 w_k1_IBM_s116
                                 + 0.78231 w_k1_INTC_s116
                                 + 0.759224 w_k1_KO_s116
                                 + 1.09022 w_k1_MAT_s116
                                 + 0.837423 w_k1_MSFT_s116
                                 + 0.872878 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.05177 w_k1_ADM_s116
                                 + 0.968335 w_k1_AXP_s116
                                 + 0.935012 w_k1_DIS_s116
                                 + 1.07055 w_k1_FDX_s116
                                 + 0.791386 w_k1_IBM_s116
                                 + 0.973803 w_k1_INTC_s116
                                 + 0.910725 w_k1_KO_s116
                                 + 0.786699 w_k1_MAT_s116
                                 + 1.03947 w_k1_MSFT_s116
                                 + 1.01619 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03663 w_k1_ADM_s116
                                 + 0.894313 w_k1_AXP_s116
                                 + 0.920461 w_k1_DIS_s116
                                 + 0.877197 w_k1_FDX_s116
                                 + 1.04767 w_k1_IBM_s116
                                 + 0.955475 w_k1_INTC_s116
                                 + 1.03633 w_k1_KO_s116 + 1.24261 w_k1_MAT_s116
                                 + 1.00587 w_k1_MSFT_s116
                                 + 1.04666 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.888706 w_k1_ADM_s117
                                 + 1.00588 w_k1_AXP_s117
                                 + 1.15074 w_k1_DIS_s117
                                 + 1.06416 w_k1_FDX_s117
                                 + 1.48772 w_k1_IBM_s117
                                 + 1.17855 w_k1_INTC_s117
                                 + 1.26637 w_k1_KO_s117 + 1.20931 w_k1_MAT_s117
                                 + 0.992696 w_k1_MSFT_s117
                                 + 0.965067 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.31439 w_k1_ADM_s117
                                 + 1.4465 w_k1_AXP_s117 + 1.65758 w_k1_DIS_s117
                                 + 1.35974 w_k1_FDX_s117
                                 + 1.60337 w_k1_IBM_s117
                                 + 1.63587 w_k1_INTC_s117
                                 + 1.57764 w_k1_KO_s117 + 1.30971 w_k1_MAT_s117
                                 + 1.29893 w_k1_MSFT_s117
                                 + 1.27645 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.02714 w_k1_ADM_s117
                                 + 0.976011 w_k1_AXP_s117
                                 + 1.05942 w_k1_DIS_s117
                                 + 1.09528 w_k1_FDX_s117
                                 + 0.909136 w_k1_IBM_s117
                                 + 1.07655 w_k1_INTC_s117
                                 + 0.945693 w_k1_KO_s117
                                 + 0.812791 w_k1_MAT_s117
                                 + 0.899141 w_k1_MSFT_s117
                                 + 1.02191 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.959247 w_k1_ADM_s117
                                 + 0.836656 w_k1_AXP_s117
                                 + 0.828373 w_k1_DIS_s117
                                 + 0.951231 w_k1_FDX_s117
                                 + 1.0761 w_k1_IBM_s117
                                 + 1.05911 w_k1_INTC_s117
                                 + 1.02321 w_k1_KO_s117
                                 + 0.987072 w_k1_MAT_s117
                                 + 0.858744 w_k1_MSFT_s117
                                 + 0.893768 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.811447 w_k1_ADM_s117
                                 + 0.700588 w_k1_AXP_s117
                                 + 0.640083 w_k1_DIS_s117
                                 + 0.778285 w_k1_FDX_s117
                                 + 0.850512 w_k1_IBM_s117
                                 + 0.826919 w_k1_INTC_s117
                                 + 0.772093 w_k1_KO_s117
                                 + 0.874718 w_k1_MAT_s117
                                 + 0.736792 w_k1_MSFT_s117
                                 + 0.765076 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.679 w_k1_ADM_s117
                                 + 1.3126 w_k1_AXP_s117 + 1.23223 w_k1_DIS_s117
                                 + 1.33992 w_k1_FDX_s117
                                 + 0.917416 w_k1_IBM_s117
                                 + 1.30574 w_k1_INTC_s117
                                 + 1.20122 w_k1_KO_s117 + 1.08503 w_k1_MAT_s117
                                 + 1.12889 w_k1_MSFT_s117
                                 + 1.13036 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.852901 w_k1_ADM_s117
                                 + 0.791036 w_k1_AXP_s117
                                 + 0.832745 w_k1_DIS_s117
                                 + 0.991768 w_k1_FDX_s117
                                 + 0.666722 w_k1_IBM_s117
                                 + 1.09118 w_k1_INTC_s117
                                 + 0.833358 w_k1_KO_s117
                                 + 0.918408 w_k1_MAT_s117
                                 + 1.01756 w_k1_MSFT_s117
                                 + 0.901702 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.68911 w_k1_ADM_s117
                                 + 1.042 w_k1_AXP_s117 + 1.05207 w_k1_DIS_s117
                                 + 1.14951 w_k1_FDX_s117 + 1.2029 w_k1_IBM_s117
                                 + 0.936502 w_k1_INTC_s117
                                 + 1.05639 w_k1_KO_s117 + 1.24371 w_k1_MAT_s117
                                 + 0.926696 w_k1_MSFT_s117
                                 + 1.02104 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0252174274084692 w_k1_ADM_s000
                                 + 0.0215598720972841 w_k1_ADM_s001
                                 + 0.100174189168551 w_k1_ADM_s002
                                 + 0.0379010269627337 w_k1_ADM_s003
                                 + 0.0289094595712901 w_k1_ADM_s004
                                 + 0.0296284936509069 w_k1_ADM_s005
                                 + 0.0320235444104996 w_k1_ADM_s006
                                 + 0.0511204496100246 w_k1_ADM_s007
                                 + 0.025483165219976 w_k1_ADM_s010
                                 + 0.070505430391394 w_k1_ADM_s011
                                 + 0.0885526358542074 w_k1_ADM_s012
                                 + 0.0270623390435032 w_k1_ADM_s013
                                 + 0.0275172082982243 w_k1_ADM_s014
                                 + 0.0668084830470453 w_k1_ADM_s015
                                 + 0.0540445803299848 w_k1_ADM_s016
                                 + 0.0558380860816497 w_k1_ADM_s017
                                 + 0.0317351726270106 w_k1_ADM_s100
                                 + 0.0106740724253239 w_k1_ADM_s101
                                 + 0.00669092849727409 w_k1_ADM_s102
                                 + 0.0144372395344935 w_k1_ADM_s103
                                 + 0.0166160848635113 w_k1_ADM_s104
                                 + 0.0258393604283939 w_k1_ADM_s105
                                 + 0.0168084941938174 w_k1_ADM_s106
                                 + 0.0216692399057756 w_k1_ADM_s107
                                 + 0.0165381520377755 w_k1_ADM_s110
                                 + 0.00679686276638712 w_k1_ADM_s111
                                 + 0.00646177887305253 w_k1_ADM_s112
                                 + 0.03306414508657 w_k1_ADM_s113
                                 + 0.00781157223387082 w_k1_ADM_s114
                                 + 0.0174911247236002 w_k1_ADM_s115
                                 + 0.0272733147031953 w_k1_ADM_s116
                                 + 0.00982727248198986 w_k1_ADM_s117
                                 + 0.0248148877657353 w_k1_AXP_s000
                                 + 0.0233436467616503 w_k1_AXP_s001
                                 + 0.118482768429305 w_k1_AXP_s002
                                 + 0.0350773343890902 w_k1_AXP_s003
                                 + 0.0271881017684647 w_k1_AXP_s004
                                 + 0.0347864560152048 w_k1_AXP_s005
                                 + 0.0315079436763297 w_k1_AXP_s006
                                 + 0.0498513612059478 w_k1_AXP_s007
                                 + 0.027583824784237 w_k1_AXP_s010
                                 + 0.0619518280202989 w_k1_AXP_s011
                                 + 0.0906587694156438 w_k1_AXP_s012
                                 + 0.0299084116099912 w_k1_AXP_s013
                                 + 0.0281518586371398 w_k1_AXP_s014
                                 + 0.0570246256493294 w_k1_AXP_s015
                                 + 0.0501842276339622 w_k1_AXP_s016
                                 + 0.0555256071149518 w_k1_AXP_s017
                                 + 0.0269508308902377 w_k1_AXP_s100
                                 + 0.0118039208553748 w_k1_AXP_s101
                                 + 0.00580578228769482 w_k1_AXP_s102
                                 + 0.0143052122401935 w_k1_AXP_s103
                                 + 0.0198971902055504 w_k1_AXP_s104
                                 + 0.0269713345364384 w_k1_AXP_s105
                                 + 0.017169621129385 w_k1_AXP_s106
                                 + 0.0221564631778451 w_k1_AXP_s107
                                 + 0.0164345917771122 w_k1_AXP_s110
                                 + 0.00818737970924391 w_k1_AXP_s111
                                 + 0.00766994723715153 w_k1_AXP_s112
                                 + 0.0324249293683306 w_k1_AXP_s113
                                 + 0.0068948078201173 w_k1_AXP_s114
                                 + 0.0175726632490902 w_k1_AXP_s115
                                 + 0.0259721353396217 w_k1_AXP_s116
                                 + 0.00838222387658433 w_k1_AXP_s117
                                 + 0.0259412417112474 w_k1_DIS_s000
                                 + 0.023786045045321 w_k1_DIS_s001
                                 + 0.111784317668576 w_k1_DIS_s002
                                 + 0.0374021135912093 w_k1_DIS_s003
                                 + 0.0276591292642968 w_k1_DIS_s004
                                 + 0.0343284697859286 w_k1_DIS_s005
                                 + 0.0327016109801518 w_k1_DIS_s006
                                 + 0.0574999118104946 w_k1_DIS_s007
                                 + 0.0264129790154928 w_k1_DIS_s010
                                 + 0.0627286341638925 w_k1_DIS_s011
                                 + 0.0915771611712438 w_k1_DIS_s012
                                 + 0.0321558150085322 w_k1_DIS_s013
                                 + 0.0270277033509504 w_k1_DIS_s014
                                 + 0.0685231819213002 w_k1_DIS_s015
                                 + 0.0481542315964488 w_k1_DIS_s016
                                 + 0.0570984239994092 w_k1_DIS_s017
                                 + 0.0263692299851226 w_k1_DIS_s100
                                 + 0.0118282476105244 w_k1_DIS_s101
                                 + 0.0069525608577064 w_k1_DIS_s102
                                 + 0.0153189213524026 w_k1_DIS_s103
                                 + 0.0201710021530021 w_k1_DIS_s104
                                 + 0.0255601289707936 w_k1_DIS_s105
                                 + 0.0173886783787944 w_k1_DIS_s106
                                 + 0.0212518090432351 w_k1_DIS_s107
                                 + 0.0163620700342258 w_k1_DIS_s110
                                 + 0.00811792917688382 w_k1_DIS_s111
                                 + 0.00720814128187262 w_k1_DIS_s112
                                 + 0.0302430768764637 w_k1_DIS_s113
                                 + 0.00636329045132208 w_k1_DIS_s114
                                 + 0.0174831055904236 w_k1_DIS_s115
                                 + 0.0239957637438472 w_k1_DIS_s116
                                 + 0.00867516002678196 w_k1_DIS_s117
                                 + 0.0254446123153077 w_k1_FDX_s000
                                 + 0.0224104521356714 w_k1_FDX_s001
                                 + 0.10980090259575 w_k1_FDX_s002
                                 + 0.035660424601303 w_k1_FDX_s003
                                 + 0.0276580693569596 w_k1_FDX_s004
                                 + 0.0328154323681941 w_k1_FDX_s005
                                 + 0.0306562364096041 w_k1_FDX_s006
                                 + 0.05384212479045 w_k1_FDX_s007
                                 + 0.0272458635456991 w_k1_FDX_s010
                                 + 0.0648047078562977 w_k1_FDX_s011
                                 + 0.0893542399824112 w_k1_FDX_s012
                                 + 0.0303812700395745 w_k1_FDX_s013
                                 + 0.0288264382062156 w_k1_FDX_s014
                                 + 0.0667834045452088 w_k1_FDX_s015
                                 + 0.0494499292429327 w_k1_FDX_s016
                                 + 0.0569679853340519 w_k1_FDX_s017
                                 + 0.0265570204129479 w_k1_FDX_s100
                                 + 0.0117822672756781 w_k1_FDX_s101
                                 + 0.00662055439092126 w_k1_FDX_s102
                                 + 0.0151460734348616 w_k1_FDX_s103
                                 + 0.0197574996910126 w_k1_FDX_s104
                                 + 0.0265279917086699 w_k1_FDX_s105
                                 + 0.0170901260748978 w_k1_FDX_s106
                                 + 0.02087816027304 w_k1_FDX_s107
                                 + 0.0152610887352828 w_k1_FDX_s110
                                 + 0.008727720277845 w_k1_FDX_s111
                                 + 0.00721081975044546 w_k1_FDX_s112
                                 + 0.0305845464023502 w_k1_FDX_s113
                                 + 0.00674388972902182 w_k1_FDX_s114
                                 + 0.0183478528472415 w_k1_FDX_s115
                                 + 0.0247584123937438 w_k1_FDX_s116
                                 + 0.00918027146475854 w_k1_FDX_s117
                                 + 0.0267148565740758 w_k1_IBM_s000
                                 + 0.0232304085265744 w_k1_IBM_s001
                                 + 0.108832054684284 w_k1_IBM_s002
                                 + 0.0388365753753001 w_k1_IBM_s003
                                 + 0.0276035140504726 w_k1_IBM_s004
                                 + 0.0338584197561433 w_k1_IBM_s005
                                 + 0.031023249033615 w_k1_IBM_s006
                                 + 0.0546565151461556 w_k1_IBM_s007
                                 + 0.0265927556025408 w_k1_IBM_s010
                                 + 0.0641392154291017 w_k1_IBM_s011
                                 + 0.0932905131854407 w_k1_IBM_s012
                                 + 0.0308608969554671 w_k1_IBM_s013
                                 + 0.02798767756453 w_k1_IBM_s014
                                 + 0.0664859590343629 w_k1_IBM_s015
                                 + 0.050023803132048 w_k1_IBM_s016
                                 + 0.0573563509178908 w_k1_IBM_s017
                                 + 0.0274368952535562 w_k1_IBM_s100
                                 + 0.0125185063027956 w_k1_IBM_s101
                                 + 0.00691932932519551 w_k1_IBM_s102
                                 + 0.0144434247216049 w_k1_IBM_s103
                                 + 0.019277544631341 w_k1_IBM_s104
                                 + 0.0261694176273148 w_k1_IBM_s105
                                 + 0.0186307557593519 w_k1_IBM_s106
                                 + 0.0218789635844193 w_k1_IBM_s107
                                 + 0.0159857869380115 w_k1_IBM_s110
                                 + 0.00845096752242891 w_k1_IBM_s111
                                 + 0.00720931110400529 w_k1_IBM_s112
                                 + 0.0309101781157066 w_k1_IBM_s113
                                 + 0.00661406595178316 w_k1_IBM_s114
                                 + 0.0185856895933461 w_k1_IBM_s115
                                 + 0.025276686287707 w_k1_IBM_s116
                                 + 0.0086039618989259 w_k1_IBM_s117
                                 + 0.0205137027501599 w_k1_INTC_s000
                                 + 0.0220767591763548 w_k1_INTC_s001
                                 + 0.107400293448704 w_k1_INTC_s002
                                 + 0.0378310951752918 w_k1_INTC_s003
                                 + 0.0256696439123655 w_k1_INTC_s004
                                 + 0.0318631076574484 w_k1_INTC_s005
                                 + 0.0325002854213461 w_k1_INTC_s006
                                 + 0.0540834572119802 w_k1_INTC_s007
                                 + 0.0250385811066679 w_k1_INTC_s010
                                 + 0.0677970791642429 w_k1_INTC_s011
                                 + 0.0952071940903171 w_k1_INTC_s012
                                 + 0.0298860489217148 w_k1_INTC_s013
                                 + 0.0287748553514314 w_k1_INTC_s014
                                 + 0.0710474160784029 w_k1_INTC_s015
                                 + 0.0506701891157243 w_k1_INTC_s016
                                 + 0.0538915043968308 w_k1_INTC_s017
                                 + 0.0286274215099125 w_k1_INTC_s100
                                 + 0.0111408575335977 w_k1_INTC_s101
                                 + 0.0077951788824887 w_k1_INTC_s102
                                 + 0.0157883335988697 w_k1_INTC_s103
                                 + 0.0200231472939087 w_k1_INTC_s104
                                 + 0.0254541729340061 w_k1_INTC_s105
                                 + 0.0148216335767453 w_k1_INTC_s106
                                 + 0.0207827552867697 w_k1_INTC_s107
                                 + 0.0155189343587973 w_k1_INTC_s110
                                 + 0.00850876523356018 w_k1_INTC_s111
                                 + 0.00672245122722135 w_k1_INTC_s112
                                 + 0.032546500449383 w_k1_INTC_s113
                                 + 0.0066949472015772 w_k1_INTC_s114
                                 + 0.0172397230395654 w_k1_INTC_s115
                                 + 0.0270294323481123 w_k1_INTC_s116
                                 + 0.00937212426932325 w_k1_INTC_s117
                                 + 0.0255126243492681 w_k1_KO_s000
                                 + 0.0210684082660352 w_k1_KO_s001
                                 + 0.117720802534344 w_k1_KO_s002
                                 + 0.0342906181481909 w_k1_KO_s003
                                 + 0.0274036370665282 w_k1_KO_s004
                                 + 0.0357959655748696 w_k1_KO_s005
                                 + 0.0350365223325156 w_k1_KO_s006
                                 + 0.0522086453145839 w_k1_KO_s007
                                 + 0.0270085510745518 w_k1_KO_s010
                                 + 0.0645331894765612 w_k1_KO_s011
                                 + 0.0879433632789649 w_k1_KO_s012
                                 + 0.0276888680175035 w_k1_KO_s013
                                 + 0.0274280293722922 w_k1_KO_s014
                                 + 0.0741320672901085 w_k1_KO_s015
                                 + 0.0519548606179298 w_k1_KO_s016
                                 + 0.0569076536353844 w_k1_KO_s017
                                 + 0.0264711027727104 w_k1_KO_s100
                                 + 0.0120706224588244 w_k1_KO_s101
                                 + 0.00656911135879669 w_k1_KO_s102
                                 + 0.0143354018986698 w_k1_KO_s103
                                 + 0.0192036725892647 w_k1_KO_s104
                                 + 0.0240231480994132 w_k1_KO_s105
                                 + 0.0177816444437662 w_k1_KO_s106
                                 + 0.0209664666969843 w_k1_KO_s107
                                 + 0.0153226892011863 w_k1_KO_s110
                                 + 0.00813796550715631 w_k1_KO_s111
                                 + 0.00644353737622095 w_k1_KO_s112
                                 + 0.0280442033787622 w_k1_KO_s113
                                 + 0.00691911732263273 w_k1_KO_s114
                                 + 0.0186272106463264 w_k1_KO_s115
                                 + 0.0248237780548992 w_k1_KO_s116
                                 + 0.00878295186925648 w_k1_KO_s117
                                 + 0.0297692334789373 w_k1_MAT_s000
                                 + 0.0257759322145873 w_k1_MAT_s001
                                 + 0.142653301727011 w_k1_MAT_s002
                                 + 0.0372857558278435 w_k1_MAT_s003
                                 + 0.030081038572136 w_k1_MAT_s004
                                 + 0.0333290305144258 w_k1_MAT_s005
                                 + 0.0293392374678687 w_k1_MAT_s006
                                 + 0.0456868520902753 w_k1_MAT_s007
                                 + 0.0356336298466439 w_k1_MAT_s010
                                 + 0.0592116688303975 w_k1_MAT_s011
                                 + 0.0986994214771811 w_k1_MAT_s012
                                 + 0.0259434437805417 w_k1_MAT_s013
                                 + 0.0288799152134983 w_k1_MAT_s014
                                 + 0.0609435157880984 w_k1_MAT_s015
                                 + 0.0538018292998002 w_k1_MAT_s016
                                 + 0.0513306219383749 w_k1_MAT_s017
                                 + 0.0225451891822276 w_k1_MAT_s100
                                 + 0.0107162716880436 w_k1_MAT_s101
                                 + 0.00581930595183008 w_k1_MAT_s102
                                 + 0.0127481286413459 w_k1_MAT_s103
                                 + 0.0168520291246072 w_k1_MAT_s104
                                 + 0.0319989242720479 w_k1_MAT_s105
                                 + 0.0158173540030123 w_k1_MAT_s106
                                 + 0.029299484561781 w_k1_MAT_s107
                                 + 0.0162508073889069 w_k1_MAT_s110
                                 + 0.00812794289931675 w_k1_MAT_s111
                                 + 0.00839908731771623 w_k1_MAT_s112
                                 + 0.0292213012628593 w_k1_MAT_s113
                                 + 0.00590980612710688 w_k1_MAT_s114
                                 + 0.0213442258801201 w_k1_MAT_s115
                                 + 0.0251645992325321 w_k1_MAT_s116
                                 + 0.00863191724513348 w_k1_MAT_s117
                                 + 0.0238660260350068 w_k1_MSFT_s000
                                 + 0.0221304339368848 w_k1_MSFT_s001
                                 + 0.107812729900655 w_k1_MSFT_s002
                                 + 0.036611416259793 w_k1_MSFT_s003
                                 + 0.027572751759685 w_k1_MSFT_s004
                                 + 0.0319279010831675 w_k1_MSFT_s005
                                 + 0.0340155849602848 w_k1_MSFT_s006
                                 + 0.0542443562241485 w_k1_MSFT_s007
                                 + 0.0277472574168514 w_k1_MSFT_s010
                                 + 0.0698017593571515 w_k1_MSFT_s011
                                 + 0.0857770297691342 w_k1_MSFT_s012
                                 + 0.0291238115884867 w_k1_MSFT_s013
                                 + 0.0266875843128853 w_k1_MSFT_s014
                                 + 0.0680851737193379 w_k1_MSFT_s015
                                 + 0.0517981584776501 w_k1_MSFT_s016
                                 + 0.060246927654431 w_k1_MSFT_s017
                                 + 0.0254227910021553 w_k1_MSFT_s100
                                 + 0.0101903954442368 w_k1_MSFT_s101
                                 + 0.00691367636714141 w_k1_MSFT_s102
                                 + 0.0158437815231939 w_k1_MSFT_s103
                                 + 0.0190994045039975 w_k1_MSFT_s104
                                 + 0.0261388422477211 w_k1_MSFT_s105
                                 + 0.0154866828624472 w_k1_MSFT_s106
                                 + 0.0220185265209734 w_k1_MSFT_s107
                                 + 0.0148099879420495 w_k1_MSFT_s110
                                 + 0.00845744823868792 w_k1_MSFT_s111
                                 + 0.00614209945664567 w_k1_MSFT_s112
                                 + 0.0308237232995739 w_k1_MSFT_s113
                                 + 0.00659830241239139 w_k1_MSFT_s114
                                 + 0.0163995439709006 w_k1_MSFT_s115
                                 + 0.0261341861886162 w_k1_MSFT_s116
                                 + 0.00818391135230383 w_k1_MSFT_s117
                                 + 0.0259904499484126 w_k1_PG_s000
                                 + 0.0231435267201775 w_k1_PG_s001
                                 + 0.111477891801916 w_k1_PG_s002
                                 + 0.0360313322325254 w_k1_PG_s003
                                 + 0.0272205948798755 w_k1_PG_s004
                                 + 0.0347516278894685 w_k1_PG_s005
                                 + 0.0317938571991726 w_k1_PG_s006
                                 + 0.0555504792327421 w_k1_PG_s007
                                 + 0.0251025575441825 w_k1_PG_s010
                                 + 0.065656506005806 w_k1_PG_s011
                                 + 0.091806437182631 w_k1_PG_s012
                                 + 0.0278206089661547 w_k1_PG_s013
                                 + 0.0283967068453325 w_k1_PG_s014
                                 + 0.0718109162929117 w_k1_PG_s015
                                 + 0.0501456291534095 w_k1_PG_s016
                                 + 0.0566994819424783 w_k1_PG_s017
                                 + 0.0256312140411385 w_k1_PG_s100
                                 + 0.0114170908655754 w_k1_PG_s101
                                 + 0.00660983522531574 w_k1_PG_s102
                                 + 0.0153155781271482 w_k1_PG_s103
                                 + 0.0190962230807523 w_k1_PG_s104
                                 + 0.0266250775254828 w_k1_PG_s105
                                 + 0.0162822280773926 w_k1_PG_s106
                                 + 0.0210455605260331 w_k1_PG_s107
                                 + 0.0151586447886544 w_k1_PG_s110
                                 + 0.00861432393453614 w_k1_PG_s111
                                 + 0.00730851580064192 w_k1_PG_s112
                                 + 0.028445586874456 w_k1_PG_s113
                                 + 0.00622136082471705 w_k1_PG_s114
                                 + 0.0186504052338213 w_k1_PG_s115
                                 + 0.0263679812513302 w_k1_PG_s116
                                 + 0.00833243612131452 w_k1_PG_s117 >= 1.08
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
