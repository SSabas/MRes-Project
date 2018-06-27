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
 w_k1_ADM_s0_return_constraint:  0.971912 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.04213 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.83585 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.08946 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.3289 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.958074 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.12914 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.952918 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.0272 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.714842 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.19088 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.32937 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.0528 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.978894 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.01651 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.827992 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1248 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.900404 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.752646 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.09571 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.36015 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.2164 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.843091 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.864352 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.94517 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.04794 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.25507 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.784014 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.806621 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.16343 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.026 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.861663 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.30404 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.804847 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.671568 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.987454 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.20889 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.099 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.940665 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.12619 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.761286 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.21252 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.28506 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.898637 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.27968 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.18931 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.976831 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.733745 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.57513 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.22322 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.924623 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.92644 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.853134 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.819893 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.0138 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.930887 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.592668 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.03305 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.30065 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.71233 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.788859 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.2197 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.94185 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.03912 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.48112 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.731372 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.543846 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.11734 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.62013 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.953397 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.61147 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.648554 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.692117 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.20163 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.11275 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.896847 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.895155 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.12092 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.75886 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.27633 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.55103 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.952297 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.24814 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.01224 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.09504 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.667249 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.58909 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.50477 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.810957 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.891418 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.874092 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.757203 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.990883 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.735636 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.596652 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.29706 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.22864 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.92291 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.06551 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.13301 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.779881 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.20843 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.40349 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.630826 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.632895 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.897812 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.4328 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.76974 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.54848 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.538109 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.573464 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.05982 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.13524 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.876833 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.968151 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.0546 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.839988 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13746 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.35259 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.890937 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.17086 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.923107 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.08673 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.726835 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.33761 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.45467 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.932834 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.01548 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.956897 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.773731 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.15832 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.857678 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.757348 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.19649 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.26036 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.43611 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.953265 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.956816 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.898614 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.08975 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.3297 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.713828 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.750438 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.10874 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.22436 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.844819 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.44262 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.771771 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.710865 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.04485 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.21791 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.06199 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.815153 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.21857 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.834753 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.17911 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.03994 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.03164 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.04471 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.54637 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.883838 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.02341 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.51705 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.822865 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.934365 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.606815 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.93381 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.3903 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.755054 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.02294 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.687753 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.859171 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.24431 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.44345 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.828488 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.57157 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.783852 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.27188 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.977039 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.769716 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.559171 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.634385 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.99201 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.20214 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.26356 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.559422 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.942816 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.37503 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.987279 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.756036 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.01224 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.02956 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.823511 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.09146 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.27678 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.919463 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.08407 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.99451 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0668 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.842703 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.25353 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.27462 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.970694 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.08604 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.908752 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.764534 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.906302 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.849694 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.733939 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.23975 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.09884 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.36068 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07119 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.952113 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.962409 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.06804 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.04778 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.663908 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.769633 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.00901 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.1373 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.919493 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.55721 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.76545 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.826046 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.23334 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.15472 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.04657 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.897807 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.15381 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.815227 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.1877 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22095 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.974056 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.12133 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.28102 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.973489 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.822128 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.53942 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.21623 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.928112 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.811269 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.894082 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.89122 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.813695 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.873237 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.712326 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.08466 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.11362 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.40683 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.982927 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.18206 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.853021 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.16291 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.19872 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.692233 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.653738 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.75664 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.51382 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.926722 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.38583 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.556262 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.76984 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.06753 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.04156 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.899916 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.87817 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.27046 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.751561 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.15732 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.14467 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.977672 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.4202 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.66587 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.915498 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.930117 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.50714 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.845221 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.803251 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.715737 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.803467 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.39892 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.781405 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.20052 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.782427 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.844711 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.50376 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.52097 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.812382 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.81027 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.76418 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.13906 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.16776 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.616221 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.485014 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.618701 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 2.03323 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.03839 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.4582 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.422789 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.764965 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.20758 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.971686 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.660308 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.981523 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.06678 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.817022 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.16702 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.34669 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.916115 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.12963 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.909744 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.09139 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.824458 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.31131 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.31679 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.941635 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.088 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.892002 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.759537 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.957731 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.822108 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.756219 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.11741 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.1522 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.29998 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.08997 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.929825 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.831723 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.960686 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.18344 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.69446 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.752256 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.881064 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.20917 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.768067 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.41419 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.734183 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.719611 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.01077 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.183 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.970519 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.913357 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.09105 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.890022 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12572 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.12602 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.884872 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.07099 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04597 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.0133 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.814138 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.22872 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.18369 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.957598 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.972329 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.934618 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.886227 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.07333 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.961299 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.816767 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.13851 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.21434 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.30784 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.935757 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.15535 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.918139 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.08351 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.2241 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.756796 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.860787 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.91668 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.39654 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.03085 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.27237 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.851772 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.853669 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.08028 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.08665 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.01823 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.101999999065085 z_k1_s0000
                                 - 0.252999994651632 z_k1_s0001
                                 - 0.123000107931454 z_k1_s0002
                                 - 0.360000541994557 z_k1_s0003
                                 - 0.128999962464657 z_k1_s0004
                                 - 0.235000740052101 z_k1_s0005
                                 - 0.134000047242379 z_k1_s0006
                                 - 0.208999928407853 z_k1_s0007
                                 - 0.375999198158457 z_k1_s0010
                                 - 0.191999985991702 z_k1_s0011
                                 - 0.448001169381092 z_k1_s0012
                                 - 0.430999316953001 z_k1_s0013
                                 - 0.169999938473884 z_k1_s0014
                                 - 0.513001561956558 z_k1_s0015
                                 - 0.385000831561811 z_k1_s0016
                                 - 0.259000155923247 z_k1_s0017
                                 - 0.214999403403742 z_k1_s0020
                                 - 0.187999973685767 z_k1_s0021
                                 - 0.145000093005019 z_k1_s0022
                                 - 0.400000906565791 z_k1_s0023
                                 - 0.174000073906009 z_k1_s0024
                                 - 0.116000112104028 z_k1_s0025
                                 - 0.486999341268248 z_k1_s0026
                                 - 0.159999985626187 z_k1_s0027
                                 - 0.673000767294888 z_k1_s0030
                                 - 0.756002765940282 z_k1_s0031
                                 - 0.940998450192103 z_k1_s0032
                                 - 0.527000615655888 z_k1_s0033
                                 - 0.312000160127066 z_k1_s0034
                                 - 0.612999202346328 z_k1_s0035
                                 - 0.673997967975436 z_k1_s0036
                                 - 0.490000481604843 z_k1_s0037
                                 - 0.284000832112091 z_k1_s0040
                                 - 0.241000493756911 z_k1_s0041
                                 - 0.162000107565037 z_k1_s0042
                                 - 0.161000219919669 z_k1_s0043
                                 - 0.476999395285164 z_k1_s0044
                                 - 0.210000099547237 z_k1_s0045
                                 - 0.471999957058325 z_k1_s0046
                                 - 0.147000069683066 z_k1_s0047
                                 - 0.341999326847803 z_k1_s0050
                                 - 0.203999889628534 z_k1_s0051
                                 - 0.277999614572202 z_k1_s0052
                                 - 0.1069999511483 z_k1_s0053
                                 - 0.137999990419307 z_k1_s0054
                                 - 0.311999288843604 z_k1_s0055
                                 - 0.0919999321462003 z_k1_s0056
                                 - 0.370999916785234 z_k1_s0057
                                 - 0.340000520043004 z_k1_s0060
                                 - 0.246000144895515 z_k1_s0061
                                 - 0.803001112533766 z_k1_s0062
                                 - 0.435001078511448 z_k1_s0063
                                 - 0.209000104829092 z_k1_s0064
                                 - 0.479001652623565 z_k1_s0065
                                 - 0.712999597667622 z_k1_s0066
                                 - 0.521001860093772 z_k1_s0067
                                 - 0.289999460029268 z_k1_s0070
                                 - 0.155000040188129 z_k1_s0071
                                 - 0.137000053975963 z_k1_s0072
                                 - 0.375000189503557 z_k1_s0073
                                 - 0.170000124048287 z_k1_s0074
                                 - 0.255000281422452 z_k1_s0075
                                 - 0.197000770816681 z_k1_s0076
                                 - 0.328000257288464 z_k1_s0077
                                 - 0.516001021497558 z_k1_s0100
                                 - 0.605003152002069 z_k1_s0101
                                 - 0.572000085068812 z_k1_s0102
                                 - 0.308000639653642 z_k1_s0103
                                 - 0.327000563187235 z_k1_s0104
                                 - 0.969001890225008 z_k1_s0105
                                 - 1.07600130722967 z_k1_s0106
                                 - 0.452001126699881 z_k1_s0107
                                 - 0.438998309055903 z_k1_s0110
                                 - 0.686001164249935 z_k1_s0111
                                 - 0.262999933883477 z_k1_s0112
                                 - 0.224000012236852 z_k1_s0113
                                 - 0.919000811705369 z_k1_s0114
                                 - 0.237999752976753 z_k1_s0115
                                 - 0.472000669655982 z_k1_s0116
                                 - 0.225999777085484 z_k1_s0117
                                 - 0.331001051865622 z_k1_s0120
                                 - 0.394001612922623 z_k1_s0121
                                 - 1.08100229431476 z_k1_s0122
                                 - 0.374002048191951 z_k1_s0123
                                 - 0.273000908942759 z_k1_s0124
                                 - 0.220000481552073 z_k1_s0125
                                 - 0.511002403956352 z_k1_s0126
                                 - 0.329000744091576 z_k1_s0127
                                 - 0.379999059285586 z_k1_s0130
                                 - 0.377000004051201 z_k1_s0131
                                 - 0.342001056655907 z_k1_s0132
                                 - 0.257999776309443 z_k1_s0133
                                 - 0.314998683552841 z_k1_s0134
                                 - 0.208999890456832 z_k1_s0135
                                 - 0.619999609991598 z_k1_s0136
                                 - 0.217999775158388 z_k1_s0137
                                 - 0.426001929162459 z_k1_s0140
                                 - 0.341999460421159 z_k1_s0141
                                 - 0.231000611023664 z_k1_s0142
                                 - 0.328000078633428 z_k1_s0143
                                 - 0.196000612050607 z_k1_s0144
                                 - 0.968002266288927 z_k1_s0145
                                 - 0.217000611434441 z_k1_s0146
                                 - 0.381001268552646 z_k1_s0147
                                 - 0.521001092354634 z_k1_s0150
                                 - 0.259999899698825 z_k1_s0151
                                 - 0.275999922427959 z_k1_s0152
                                 - 0.413000469283374 z_k1_s0153
                                 - 0.428999865813035 z_k1_s0154
                                 - 0.426999158497487 z_k1_s0155
                                 - 0.483000177348665 z_k1_s0156
                                 - 0.322001161898232 z_k1_s0157
                                 - 0.44599873722144 z_k1_s0160
                                 - 0.317000233407531 z_k1_s0161
                                 - 0.381000516758211 z_k1_s0162
                                 - 0.241000260679004 z_k1_s0163
                                 - 0.294000343766117 z_k1_s0164
                                 - 0.326000291441241 z_k1_s0165
                                 - 1.16900262326191 z_k1_s0166
                                 - 0.706001707085604 z_k1_s0167
                                 - 0.479000844404898 z_k1_s0170
                                 - 0.639998699401976 z_k1_s0171
                                 - 0.276999769925425 z_k1_s0172
                                 - 0.344998914849639 z_k1_s0173
                                 - 0.276000224279324 z_k1_s0174
                                 - 0.294999784552337 z_k1_s0175
                                 - 0.256999844706601 z_k1_s0176
                                 - 0.161999858542309 z_k1_s0177
                                 - 0.169000005747003 z_k1_s1000
                                 - 0.0519999404298704 z_k1_s1001
                                 - 0.0619999528279133 z_k1_s1002
                                 - 0.0949995418816268 z_k1_s1003
                                 - 0.212999635018554 z_k1_s1004
                                 - 0.0749999600853725 z_k1_s1005
                                 - 0.151000036590506 z_k1_s1006
                                 - 0.06899997036654 z_k1_s1007
                                 - 0.616999302523407 z_k1_s1010
                                 - 0.128999937612322 z_k1_s1011
                                 - 0.134999964791956 z_k1_s1012
                                 - 0.505000347369755 z_k1_s1013
                                 - 0.299000190302098 z_k1_s1014
                                 - 0.434998736774448 z_k1_s1015
                                 - 0.297999021622531 z_k1_s1016
                                 - 0.16899986010996 z_k1_s1017
                                 - 0.30600043685304 z_k1_s1020
                                 - 0.320000387637326 z_k1_s1021
                                 - 0.382998842669655 z_k1_s1022
                                 - 0.235999359434659 z_k1_s1023
                                 - 0.475998326665699 z_k1_s1024
                                 - 0.212999629360041 z_k1_s1025
                                 - 0.40900026383699 z_k1_s1026
                                 - 0.303999309457896 z_k1_s1027
                                 - 0.567999699682499 z_k1_s1030
                                 - 0.267998337262201 z_k1_s1031
                                 - 0.399998304568611 z_k1_s1032
                                 - 0.348998915835189 z_k1_s1033
                                 - 0.344997879191967 z_k1_s1034
                                 - 0.201999407206532 z_k1_s1035
                                 - 0.30399976506968 z_k1_s1036
                                 - 0.197999424160846 z_k1_s1037
                                 - 0.345001527220187 z_k1_s1040
                                 - 0.185000282282126 z_k1_s1041
                                 - 0.220000585887395 z_k1_s1042
                                 - 0.518000189908654 z_k1_s1043
                                 - 0.241000417769797 z_k1_s1044
                                 - 0.612000533498924 z_k1_s1045
                                 - 0.236000410140551 z_k1_s1046
                                 - 0.145000221248153 z_k1_s1047
                                 - 0.349000893180467 z_k1_s1050
                                 - 0.0710000208518042 z_k1_s1051
                                 - 0.120000016480503 z_k1_s1052
                                 - 0.343999806197298 z_k1_s1053
                                 - 0.226999490406129 z_k1_s1054
                                 - 0.34999996877642 z_k1_s1055
                                 - 0.234000631582338 z_k1_s1056
                                 - 0.207999927072811 z_k1_s1057
                                 - 0.152000351182966 z_k1_s1060
                                 - 0.0560000874673613 z_k1_s1061
                                 - 0.110000339882506 z_k1_s1062
                                 - 0.0530000452887104 z_k1_s1063
                                 - 0.116000134639482 z_k1_s1064
                                 - 0.0620000994245814 z_k1_s1065
                                 - 0.124000343649326 z_k1_s1066
                                 - 0.0510000654363308 z_k1_s1067
                                 - 0.0379999995608942 z_k1_s1070
                                 - 0.0929997732359668 z_k1_s1071
                                 - 0.11200011731627 z_k1_s1072
                                 - 0.0569999680413544 z_k1_s1073
                                 - 0.0479999697927209 z_k1_s1074
                                 - 0.0969999115686349 z_k1_s1075
                                 - 0.0670001260731028 z_k1_s1076
                                 - 0.114000186482604 z_k1_s1077
                                 - 0.471999888547285 z_k1_s1100
                                 - 0.201000138283107 z_k1_s1101
                                 - 0.56499947390502 z_k1_s1102
                                 - 0.508001151589347 z_k1_s1103
                                 - 0.435001287254733 z_k1_s1104
                                 - 0.143000064715834 z_k1_s1105
                                 - 0.22900004191555 z_k1_s1106
                                 - 0.389000163701111 z_k1_s1107
                                 - 0.396999556761223 z_k1_s1110
                                 - 0.193000099002924 z_k1_s1111
                                 - 0.534000561026847 z_k1_s1112
                                 - 0.270000169392185 z_k1_s1113
                                 - 0.422000968751743 z_k1_s1114
                                 - 0.667999572212953 z_k1_s1115
                                 - 0.344000000202558 z_k1_s1116
                                 - 0.679001035169085 z_k1_s1117
                                 - 0.674996137400748 z_k1_s1120
                                 - 0.471997724766283 z_k1_s1121
                                 - 0.30399893173465 z_k1_s1122
                                 - 1.146998283153 z_k1_s1123
                                 - 0.383998883295662 z_k1_s1124
                                 - 0.271998951110246 z_k1_s1125
                                 - 0.28499930244402 z_k1_s1126
                                 - 0.881995542335106 z_k1_s1127
                                 - 0.303999345022864 z_k1_s1130
                                 - 0.979996454485805 z_k1_s1131
                                 - 0.74199690034047 z_k1_s1132
                                 - 0.405998669543713 z_k1_s1133
                                 - 0.431998763307072 z_k1_s1134
                                 - 0.895996896786616 z_k1_s1135
                                 - 0.403998587808289 z_k1_s1136
                                 - 0.679998726129529 z_k1_s1137
                                 - 0.202000203210193 z_k1_s1140
                                 - 0.437001502950049 z_k1_s1141
                                 - 0.396999332610846 z_k1_s1142
                                 - 0.374000166666343 z_k1_s1143
                                 - 0.154000027803778 z_k1_s1144
                                 - 0.477000203288355 z_k1_s1145
                                 - 0.710000353510981 z_k1_s1146
                                 - 0.719001535837481 z_k1_s1147
                                 - 0.342000172444183 z_k1_s1150
                                 - 0.470000082089946 z_k1_s1151
                                 - 0.38100081168787 z_k1_s1152
                                 - 0.647000945811123 z_k1_s1153
                                 - 0.380000191604648 z_k1_s1154
                                 - 0.39900020118488 z_k1_s1155
                                 - 0.188000032835979 z_k1_s1156
                                 - 0.13599986721112 z_k1_s1157
                                 - 0.805998677467653 z_k1_s1160
                                 - 0.706998995960549 z_k1_s1161
                                 - 0.670997282327821 z_k1_s1162
                                 - 0.383998825615223 z_k1_s1163
                                 - 0.771999295029143 z_k1_s1164
                                 - 0.746995310015914 z_k1_s1165
                                 - 0.498998542039342 z_k1_s1166
                                 - 0.4449984870817 z_k1_s1167
                                 - 1.36000064068274 z_k1_s1170
                                 - 0.989000510630334 z_k1_s1171
                                 - 1.74400046380481 z_k1_s1172
                                 - 0.38700037610758 z_k1_s1173
                                 - 1.65000003193103 z_k1_s1174
                                 - 1.6730008924855 z_k1_s1175
                                 - 1.33200223748686 z_k1_s1176
                                 - 1.00200020369999 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.22298 w_k1_ADM_s000
                                 + 1.28536 w_k1_AXP_s000
                                 + 0.907194 w_k1_DIS_s000
                                 + 0.941044 w_k1_FDX_s000
                                 + 1.43504 w_k1_IBM_s000
                                 + 1.12851 w_k1_INTC_s000
                                 + 1.13265 w_k1_KO_s000 + 1.06914 w_k1_MAT_s000
                                 + 0.94244 w_k1_MSFT_s000
                                 + 1.26515 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.04843 w_k1_ADM_s000
                                 + 1.02146 w_k1_AXP_s000
                                 + 1.02101 w_k1_DIS_s000
                                 + 0.971214 w_k1_FDX_s000
                                 + 1.2036 w_k1_IBM_s000
                                 + 0.945187 w_k1_INTC_s000
                                 + 1.06021 w_k1_KO_s000 + 1.0146 w_k1_MAT_s000
                                 + 0.921628 w_k1_MSFT_s000
                                 + 1.00372 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.18816 w_k1_ADM_s000
                                 + 1.47043 w_k1_AXP_s000
                                 + 1.36848 w_k1_DIS_s000
                                 + 1.21537 w_k1_FDX_s000
                                 + 1.58943 w_k1_IBM_s000
                                 + 1.16369 w_k1_INTC_s000
                                 + 1.24343 w_k1_KO_s000 + 1.31172 w_k1_MAT_s000
                                 + 1.09732 w_k1_MSFT_s000
                                 + 1.38513 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.09175 w_k1_ADM_s000
                                 + 1.19913 w_k1_AXP_s000
                                 + 1.02144 w_k1_DIS_s000
                                 + 1.07181 w_k1_FDX_s000
                                 + 0.868554 w_k1_IBM_s000
                                 + 1.05751 w_k1_INTC_s000
                                 + 0.957189 w_k1_KO_s000
                                 + 0.849926 w_k1_MAT_s000
                                 + 0.962138 w_k1_MSFT_s000
                                 + 1.02152 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.9795 w_k1_ADM_s000
                                 + 0.73223 w_k1_AXP_s000
                                 + 0.657109 w_k1_DIS_s000
                                 + 0.860947 w_k1_FDX_s000
                                 + 0.708171 w_k1_IBM_s000
                                 + 0.856589 w_k1_INTC_s000
                                 + 0.685459 w_k1_KO_s000
                                 + 0.47567 w_k1_MAT_s000
                                 + 0.744377 w_k1_MSFT_s000
                                 + 0.902454 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.876046 w_k1_ADM_s000
                                 + 0.887843 w_k1_AXP_s000
                                 + 0.637456 w_k1_DIS_s000
                                 + 0.785746 w_k1_FDX_s000
                                 + 0.927146 w_k1_IBM_s000
                                 + 0.937057 w_k1_INTC_s000
                                 + 0.788251 w_k1_KO_s000
                                 + 0.880101 w_k1_MAT_s000
                                 + 0.787786 w_k1_MSFT_s000
                                 + 0.839607 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.2063 w_k1_ADM_s000
                                 + 1.0498 w_k1_AXP_s000 + 1.30549 w_k1_DIS_s000
                                 + 1.13301 w_k1_FDX_s000
                                 + 1.04285 w_k1_IBM_s000
                                 + 1.20125 w_k1_INTC_s000
                                 + 1.13876 w_k1_KO_s000
                                 + 0.918954 w_k1_MAT_s000
                                 + 1.28975 w_k1_MSFT_s000
                                 + 1.17736 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.37444 w_k1_ADM_s000
                                 + 1.38019 w_k1_AXP_s000
                                 + 1.53847 w_k1_DIS_s000
                                 + 1.32084 w_k1_FDX_s000
                                 + 1.26043 w_k1_IBM_s000
                                 + 1.26099 w_k1_INTC_s000
                                 + 1.21767 w_k1_KO_s000
                                 + 0.832455 w_k1_MAT_s000
                                 + 1.24913 w_k1_MSFT_s000
                                 + 1.36681 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.21131 w_k1_ADM_s001
                                 + 1.11946 w_k1_AXP_s001
                                 + 1.30926 w_k1_DIS_s001
                                 + 1.14981 w_k1_FDX_s001
                                 + 1.14515 w_k1_IBM_s001
                                 + 1.08989 w_k1_INTC_s001
                                 + 1.02151 w_k1_KO_s001 + 1.11616 w_k1_MAT_s001
                                 + 1.1014 w_k1_MSFT_s001 + 1.06438 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.785596 w_k1_ADM_s001
                                 + 0.932565 w_k1_AXP_s001
                                 + 1.27101 w_k1_DIS_s001
                                 + 0.966212 w_k1_FDX_s001
                                 + 1.49209 w_k1_IBM_s001
                                 + 1.04698 w_k1_INTC_s001
                                 + 1.25757 w_k1_KO_s001 + 1.402 w_k1_MAT_s001
                                 + 1.12523 w_k1_MSFT_s001
                                 + 1.01724 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.04252 w_k1_ADM_s001
                                 + 0.809466 w_k1_AXP_s001
                                 + 0.816577 w_k1_DIS_s001
                                 + 0.918002 w_k1_FDX_s001
                                 + 0.68973 w_k1_IBM_s001
                                 + 1.0081 w_k1_INTC_s001
                                 + 0.702725 w_k1_KO_s001
                                 + 0.454335 w_k1_MAT_s001
                                 + 1.00666 w_k1_MSFT_s001
                                 + 0.938267 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.06193 w_k1_ADM_s001
                                 + 0.934328 w_k1_AXP_s001
                                 + 0.788851 w_k1_DIS_s001
                                 + 0.91256 w_k1_FDX_s001 + 1.0436 w_k1_IBM_s001
                                 + 0.970148 w_k1_INTC_s001
                                 + 0.858027 w_k1_KO_s001
                                 + 1.03084 w_k1_MAT_s001
                                 + 1.04778 w_k1_MSFT_s001
                                 + 0.989674 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.06091 w_k1_ADM_s001
                                 + 1.08934 w_k1_AXP_s001
                                 + 0.875453 w_k1_DIS_s001
                                 + 0.865065 w_k1_FDX_s001
                                 + 1.4205 w_k1_IBM_s001
                                 + 0.995049 w_k1_INTC_s001
                                 + 0.952745 w_k1_KO_s001
                                 + 1.26347 w_k1_MAT_s001
                                 + 0.993832 w_k1_MSFT_s001
                                 + 0.969993 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.34171 w_k1_ADM_s001
                                 + 1.1823 w_k1_AXP_s001 + 1.3578 w_k1_DIS_s001
                                 + 1.31359 w_k1_FDX_s001
                                 + 0.988825 w_k1_IBM_s001
                                 + 1.06033 w_k1_INTC_s001
                                 + 1.01335 w_k1_KO_s001 + 0.91963 w_k1_MAT_s001
                                 + 1.39189 w_k1_MSFT_s001 + 1.1812 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.837483 w_k1_ADM_s001
                                 + 0.607655 w_k1_AXP_s001
                                 + 0.732954 w_k1_DIS_s001
                                 + 0.735834 w_k1_FDX_s001
                                 + 0.941103 w_k1_IBM_s001
                                 + 0.830912 w_k1_INTC_s001
                                 + 0.760253 w_k1_KO_s001
                                 + 0.942993 w_k1_MAT_s001
                                 + 0.830751 w_k1_MSFT_s001
                                 + 0.873671 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.03995 w_k1_ADM_s001
                                 + 1.16835 w_k1_AXP_s001
                                 + 1.54237 w_k1_DIS_s001
                                 + 1.25136 w_k1_FDX_s001
                                 + 1.66794 w_k1_IBM_s001
                                 + 1.27216 w_k1_INTC_s001
                                 + 1.34747 w_k1_KO_s001 + 1.68437 w_k1_MAT_s001
                                 + 1.3273 w_k1_MSFT_s001 + 1.13297 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.53442 w_k1_ADM_s002
                                 + 1.54817 w_k1_AXP_s002 + 1.3657 w_k1_DIS_s002
                                 + 1.35675 w_k1_FDX_s002
                                 + 1.31085 w_k1_IBM_s002
                                 + 1.35711 w_k1_INTC_s002
                                 + 1.12497 w_k1_KO_s002
                                 + 0.993489 w_k1_MAT_s002
                                 + 1.08292 w_k1_MSFT_s002
                                 + 1.17838 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.28573 w_k1_ADM_s002
                                 + 1.19898 w_k1_AXP_s002
                                 + 0.86114 w_k1_DIS_s002
                                 + 1.05555 w_k1_FDX_s002
                                 + 1.00294 w_k1_IBM_s002
                                 + 1.00698 w_k1_INTC_s002
                                 + 1.00357 w_k1_KO_s002 + 1.1816 w_k1_MAT_s002
                                 + 1.05749 w_k1_MSFT_s002
                                 + 1.05729 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.11899 w_k1_ADM_s002
                                 + 1.20584 w_k1_AXP_s002
                                 + 1.21795 w_k1_DIS_s002
                                 + 1.15217 w_k1_FDX_s002
                                 + 1.21416 w_k1_IBM_s002
                                 + 1.0763 w_k1_INTC_s002 + 1.34067 w_k1_KO_s002
                                 + 1.29498 w_k1_MAT_s002
                                 + 1.1048 w_k1_MSFT_s002 + 1.22228 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.04765 w_k1_ADM_s002
                                 + 0.976978 w_k1_AXP_s002
                                 + 1.10507 w_k1_DIS_s002
                                 + 0.944489 w_k1_FDX_s002
                                 + 1.27211 w_k1_IBM_s002
                                 + 0.821499 w_k1_INTC_s002
                                 + 1.20157 w_k1_KO_s002 + 1.1885 w_k1_MAT_s002
                                 + 0.931404 w_k1_MSFT_s002
                                 + 1.01121 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.37892 w_k1_ADM_s002
                                 + 0.74923 w_k1_AXP_s002
                                 + 0.991895 w_k1_DIS_s002
                                 + 0.991569 w_k1_FDX_s002
                                 + 0.839322 w_k1_IBM_s002
                                 + 0.943378 w_k1_INTC_s002
                                 + 0.91211 w_k1_KO_s002
                                 + 0.487603 w_k1_MAT_s002
                                 + 1.00806 w_k1_MSFT_s002
                                 + 1.02349 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.31716 w_k1_ADM_s002
                                 + 1.33075 w_k1_AXP_s002
                                 + 0.920984 w_k1_DIS_s002
                                 + 1.19688 w_k1_FDX_s002
                                 + 0.820264 w_k1_IBM_s002
                                 + 1.01445 w_k1_INTC_s002
                                 + 0.799388 w_k1_KO_s002
                                 + 0.595542 w_k1_MAT_s002
                                 + 0.952924 w_k1_MSFT_s002
                                 + 1.01224 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.900791 w_k1_ADM_s002
                                 + 0.890881 w_k1_AXP_s002
                                 + 0.602126 w_k1_DIS_s002
                                 + 0.861366 w_k1_FDX_s002
                                 + 0.74809 w_k1_IBM_s002
                                 + 0.896122 w_k1_INTC_s002
                                 + 0.799831 w_k1_KO_s002
                                 + 0.859988 w_k1_MAT_s002
                                 + 0.79614 w_k1_MSFT_s002
                                 + 0.869356 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.34472 w_k1_ADM_s002
                                 + 1.36317 w_k1_AXP_s002 + 1.6873 w_k1_DIS_s002
                                 + 1.25562 w_k1_FDX_s002
                                 + 1.28287 w_k1_IBM_s002
                                 + 1.06025 w_k1_INTC_s002
                                 + 1.49647 w_k1_KO_s002
                                 + 0.999184 w_k1_MAT_s002
                                 + 1.34193 w_k1_MSFT_s002
                                 + 1.19184 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.757355 w_k1_ADM_s003
                                 + 0.823315 w_k1_AXP_s003
                                 + 0.742575 w_k1_DIS_s003
                                 + 0.742768 w_k1_FDX_s003
                                 + 1.09622 w_k1_IBM_s003
                                 + 0.820568 w_k1_INTC_s003
                                 + 0.986771 w_k1_KO_s003
                                 + 0.969236 w_k1_MAT_s003
                                 + 0.770004 w_k1_MSFT_s003
                                 + 1.02326 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.907922 w_k1_ADM_s003
                                 + 0.692165 w_k1_AXP_s003
                                 + 0.571268 w_k1_DIS_s003
                                 + 0.805653 w_k1_FDX_s003
                                 + 0.552543 w_k1_IBM_s003
                                 + 0.773511 w_k1_INTC_s003
                                 + 0.643585 w_k1_KO_s003
                                 + 0.618719 w_k1_MAT_s003
                                 + 0.788128 w_k1_MSFT_s003
                                 + 0.860546 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.17661 w_k1_ADM_s003
                                 + 0.984486 w_k1_AXP_s003
                                 + 1.07834 w_k1_DIS_s003
                                 + 1.12849 w_k1_FDX_s003
                                 + 0.764454 w_k1_IBM_s003
                                 + 0.923989 w_k1_INTC_s003
                                 + 0.947362 w_k1_KO_s003
                                 + 0.810342 w_k1_MAT_s003
                                 + 0.964998 w_k1_MSFT_s003
                                 + 1.1584 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.44583 w_k1_ADM_s003
                                 + 1.59056 w_k1_AXP_s003
                                 + 1.61963 w_k1_DIS_s003
                                 + 1.35329 w_k1_FDX_s003 + 1.47 w_k1_IBM_s003
                                 + 1.33991 w_k1_INTC_s003
                                 + 1.44443 w_k1_KO_s003 + 1.45559 w_k1_MAT_s003
                                 + 1.27838 w_k1_MSFT_s003
                                 + 1.40557 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.12134 w_k1_ADM_s003
                                 + 1.44176 w_k1_AXP_s003
                                 + 1.45244 w_k1_DIS_s003
                                 + 1.33128 w_k1_FDX_s003
                                 + 0.945524 w_k1_IBM_s003
                                 + 1.20889 w_k1_INTC_s003
                                 + 1.22964 w_k1_KO_s003 + 1.45757 w_k1_MAT_s003
                                 + 1.36725 w_k1_MSFT_s003
                                 + 1.35439 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.01378 w_k1_ADM_s003
                                 + 1.05568 w_k1_AXP_s003
                                 + 1.42673 w_k1_DIS_s003
                                 + 1.12758 w_k1_FDX_s003
                                 + 1.60417 w_k1_IBM_s003
                                 + 1.04726 w_k1_INTC_s003 + 1.4813 w_k1_KO_s003
                                 + 1.5419 w_k1_MAT_s003
                                 + 1.24074 w_k1_MSFT_s003 + 1.2054 w_k1_PG_s003
                                 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.17203 w_k1_ADM_s003
                                 + 0.976678 w_k1_AXP_s003
                                 + 1.10747 w_k1_DIS_s003
                                 + 0.992507 w_k1_FDX_s003
                                 + 0.928652 w_k1_IBM_s003
                                 + 0.879308 w_k1_INTC_s003
                                 + 0.968377 w_k1_KO_s003
                                 + 0.829229 w_k1_MAT_s003
                                 + 1.02922 w_k1_MSFT_s003
                                 + 1.04859 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.984941 w_k1_ADM_s003
                                 + 1.089 w_k1_AXP_s003 + 1.03858 w_k1_DIS_s003
                                 + 1.01692 w_k1_FDX_s003
                                 + 1.18851 w_k1_IBM_s003
                                 + 0.964539 w_k1_INTC_s003
                                 + 1.13761 w_k1_KO_s003 + 1.11695 w_k1_MAT_s003
                                 + 0.921967 w_k1_MSFT_s003
                                 + 1.14213 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.987264 w_k1_ADM_s004
                                 + 0.659732 w_k1_AXP_s004
                                 + 0.681367 w_k1_DIS_s004
                                 + 0.717502 w_k1_FDX_s004
                                 + 0.78805 w_k1_IBM_s004
                                 + 0.831594 w_k1_INTC_s004
                                 + 0.725293 w_k1_KO_s004
                                 + 0.821261 w_k1_MAT_s004
                                 + 0.764663 w_k1_MSFT_s004
                                 + 0.855502 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.940396 w_k1_ADM_s004
                                 + 1.14048 w_k1_AXP_s004
                                 + 0.773439 w_k1_DIS_s004
                                 + 0.847763 w_k1_FDX_s004
                                 + 1.15485 w_k1_IBM_s004
                                 + 0.893218 w_k1_INTC_s004
                                 + 0.78878 w_k1_KO_s004 + 1.57623 w_k1_MAT_s004
                                 + 0.848255 w_k1_MSFT_s004
                                 + 0.915959 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.18355 w_k1_ADM_s004
                                 + 1.08402 w_k1_AXP_s004
                                 + 1.33648 w_k1_DIS_s004
                                 + 1.20844 w_k1_FDX_s004
                                 + 0.687614 w_k1_IBM_s004
                                 + 1.14919 w_k1_INTC_s004
                                 + 0.91553 w_k1_KO_s004 + 1.08866 w_k1_MAT_s004
                                 + 1.2798 w_k1_MSFT_s004 + 1.15636 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.03396 w_k1_ADM_s004
                                 + 0.852486 w_k1_AXP_s004
                                 + 0.896638 w_k1_DIS_s004
                                 + 0.766738 w_k1_FDX_s004
                                 + 1.28269 w_k1_IBM_s004
                                 + 0.925209 w_k1_INTC_s004
                                 + 1.04256 w_k1_KO_s004 + 1.04212 w_k1_MAT_s004
                                 + 0.923166 w_k1_MSFT_s004
                                 + 1.03829 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.375 w_k1_ADM_s004
                                 + 1.15563 w_k1_AXP_s004
                                 + 1.33754 w_k1_DIS_s004
                                 + 1.18906 w_k1_FDX_s004
                                 + 1.48309 w_k1_IBM_s004
                                 + 1.30372 w_k1_INTC_s004
                                 + 1.32687 w_k1_KO_s004 + 1.85801 w_k1_MAT_s004
                                 + 1.19816 w_k1_MSFT_s004
                                 + 1.10091 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.18769 w_k1_ADM_s004
                                 + 0.784007 w_k1_AXP_s004
                                 + 1.22731 w_k1_DIS_s004
                                 + 1.09427 w_k1_FDX_s004 + 1.1103 w_k1_IBM_s004
                                 + 1.05025 w_k1_INTC_s004
                                 + 1.10356 w_k1_KO_s004
                                 + 0.999906 w_k1_MAT_s004
                                 + 1.09263 w_k1_MSFT_s004
                                 + 1.06139 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.28836 w_k1_ADM_s004
                                 + 1.05163 w_k1_AXP_s004
                                 + 0.809562 w_k1_DIS_s004
                                 + 0.995606 w_k1_FDX_s004
                                 + 0.814191 w_k1_IBM_s004
                                 + 1.12719 w_k1_INTC_s004
                                 + 0.766475 w_k1_KO_s004
                                 + 0.853943 w_k1_MAT_s004
                                 + 0.996498 w_k1_MSFT_s004
                                 + 0.986823 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.846045 w_k1_ADM_s004
                                 + 0.695058 w_k1_AXP_s004
                                 + 1.14691 w_k1_DIS_s004
                                 + 0.909753 w_k1_FDX_s004
                                 + 0.862278 w_k1_IBM_s004
                                 + 0.850516 w_k1_INTC_s004
                                 + 0.906029 w_k1_KO_s004
                                 + 0.836854 w_k1_MAT_s004
                                 + 1.02745 w_k1_MSFT_s004
                                 + 0.958231 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.945244 w_k1_ADM_s005
                                 + 0.964504 w_k1_AXP_s005
                                 + 0.78175 w_k1_DIS_s005
                                 + 0.911999 w_k1_FDX_s005
                                 + 0.714569 w_k1_IBM_s005
                                 + 0.752746 w_k1_INTC_s005
                                 + 0.768156 w_k1_KO_s005
                                 + 0.828235 w_k1_MAT_s005
                                 + 0.842049 w_k1_MSFT_s005
                                 + 0.958084 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.970285 w_k1_ADM_s005
                                 + 1.06101 w_k1_AXP_s005
                                 + 1.05671 w_k1_DIS_s005
                                 + 0.921509 w_k1_FDX_s005
                                 + 1.67977 w_k1_IBM_s005
                                 + 1.08915 w_k1_INTC_s005
                                 + 1.19882 w_k1_KO_s005 + 1.69858 w_k1_MAT_s005
                                 + 1.02991 w_k1_MSFT_s005
                                 + 0.910721 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.05363 w_k1_ADM_s005
                                 + 1.31302 w_k1_AXP_s005
                                 + 1.21876 w_k1_DIS_s005
                                 + 1.20377 w_k1_FDX_s005
                                 + 1.16867 w_k1_IBM_s005
                                 + 1.05751 w_k1_INTC_s005
                                 + 0.944017 w_k1_KO_s005
                                 + 1.29597 w_k1_MAT_s005
                                 + 1.16021 w_k1_MSFT_s005
                                 + 1.10918 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.32006 w_k1_ADM_s005
                                 + 1.44204 w_k1_AXP_s005
                                 + 1.84512 w_k1_DIS_s005
                                 + 1.43667 w_k1_FDX_s005
                                 + 1.35406 w_k1_IBM_s005
                                 + 1.19404 w_k1_INTC_s005
                                 + 1.26199 w_k1_KO_s005 + 2.39436 w_k1_MAT_s005
                                 + 1.50507 w_k1_MSFT_s005
                                 + 1.35327 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.744647 w_k1_ADM_s005
                                 + 0.699296 w_k1_AXP_s005
                                 + 0.600562 w_k1_DIS_s005
                                 + 0.670005 w_k1_FDX_s005
                                 + 0.741765 w_k1_IBM_s005
                                 + 0.766353 w_k1_INTC_s005
                                 + 0.668728 w_k1_KO_s005
                                 + 0.720628 w_k1_MAT_s005
                                 + 0.738471 w_k1_MSFT_s005
                                 + 0.774549 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.827215 w_k1_ADM_s005
                                 + 0.96514 w_k1_AXP_s005
                                 + 0.708628 w_k1_DIS_s005
                                 + 0.827204 w_k1_FDX_s005
                                 + 1.09845 w_k1_IBM_s005
                                 + 0.957273 w_k1_INTC_s005
                                 + 0.885178 w_k1_KO_s005
                                 + 1.12792 w_k1_MAT_s005
                                 + 0.936444 w_k1_MSFT_s005
                                 + 0.89187 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.08617 w_k1_ADM_s005
                                 + 1.72405 w_k1_AXP_s005
                                 + 1.48659 w_k1_DIS_s005
                                 + 1.25353 w_k1_FDX_s005
                                 + 1.39795 w_k1_IBM_s005
                                 + 1.04504 w_k1_INTC_s005
                                 + 1.22404 w_k1_KO_s005 + 2.213 w_k1_MAT_s005
                                 + 1.30762 w_k1_MSFT_s005
                                 + 1.21183 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04937 w_k1_ADM_s005
                                 + 1.05788 w_k1_AXP_s005
                                 + 1.13247 w_k1_DIS_s005
                                 + 1.02606 w_k1_FDX_s005
                                 + 1.03236 w_k1_IBM_s005
                                 + 1.0061 w_k1_INTC_s005
                                 + 0.958109 w_k1_KO_s005
                                 + 1.24626 w_k1_MAT_s005
                                 + 0.972108 w_k1_MSFT_s005
                                 + 1.09437 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.75075 w_k1_ADM_s006
                                 + 0.726886 w_k1_AXP_s006
                                 + 0.745122 w_k1_DIS_s006
                                 + 0.842834 w_k1_FDX_s006
                                 + 0.592293 w_k1_IBM_s006
                                 + 0.803061 w_k1_INTC_s006
                                 + 0.630319 w_k1_KO_s006
                                 + 0.536989 w_k1_MAT_s006
                                 + 0.817112 w_k1_MSFT_s006
                                 + 0.738644 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.896389 w_k1_ADM_s006
                                 + 1.31903 w_k1_AXP_s006
                                 + 1.01328 w_k1_DIS_s006
                                 + 0.938849 w_k1_FDX_s006
                                 + 1.30582 w_k1_IBM_s006
                                 + 0.818507 w_k1_INTC_s006
                                 + 0.987427 w_k1_KO_s006
                                 + 1.65602 w_k1_MAT_s006
                                 + 0.830081 w_k1_MSFT_s006
                                 + 1.02067 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.854711 w_k1_ADM_s006
                                 + 0.935129 w_k1_AXP_s006
                                 + 0.898243 w_k1_DIS_s006
                                 + 0.908395 w_k1_FDX_s006
                                 + 0.800173 w_k1_IBM_s006
                                 + 0.905927 w_k1_INTC_s006
                                 + 0.848446 w_k1_KO_s006
                                 + 1.04586 w_k1_MAT_s006
                                 + 0.877389 w_k1_MSFT_s006
                                 + 0.894995 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.99003 w_k1_ADM_s006
                                 + 1.0993 w_k1_AXP_s006 + 1.3081 w_k1_DIS_s006
                                 + 1.09401 w_k1_FDX_s006
                                 + 1.24933 w_k1_IBM_s006
                                 + 1.0949 w_k1_INTC_s006 + 1.15143 w_k1_KO_s006
                                 + 1.29805 w_k1_MAT_s006
                                 + 1.05557 w_k1_MSFT_s006
                                 + 1.20707 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.02523 w_k1_ADM_s006
                                 + 0.985126 w_k1_AXP_s006 + 1.195 w_k1_DIS_s006
                                 + 1.07973 w_k1_FDX_s006
                                 + 0.920145 w_k1_IBM_s006
                                 + 1.02163 w_k1_INTC_s006
                                 + 0.936124 w_k1_KO_s006
                                 + 1.15227 w_k1_MAT_s006
                                 + 1.20612 w_k1_MSFT_s006
                                 + 1.01313 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.20114 w_k1_ADM_s006
                                 + 1.46321 w_k1_AXP_s006
                                 + 1.28139 w_k1_DIS_s006
                                 + 1.27069 w_k1_FDX_s006
                                 + 1.03237 w_k1_IBM_s006
                                 + 1.10848 w_k1_INTC_s006
                                 + 0.951579 w_k1_KO_s006
                                 + 1.10714 w_k1_MAT_s006
                                 + 1.11558 w_k1_MSFT_s006 + 1.0241 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.9146 w_k1_ADM_s006
                                 + 0.911865 w_k1_AXP_s006
                                 + 0.74691 w_k1_DIS_s006
                                 + 0.825302 w_k1_FDX_s006
                                 + 1.00961 w_k1_IBM_s006
                                 + 0.758371 w_k1_INTC_s006
                                 + 0.831409 w_k1_KO_s006
                                 + 1.23081 w_k1_MAT_s006
                                 + 0.765116 w_k1_MSFT_s006
                                 + 0.912429 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.01472 w_k1_ADM_s006
                                 + 1.59873 w_k1_AXP_s006
                                 + 1.37926 w_k1_DIS_s006
                                 + 1.16627 w_k1_FDX_s006
                                 + 1.66911 w_k1_IBM_s006
                                 + 1.20456 w_k1_INTC_s006
                                 + 1.27515 w_k1_KO_s006 + 1.87586 w_k1_MAT_s006
                                 + 1.08189 w_k1_MSFT_s006
                                 + 1.10579 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.902348 w_k1_ADM_s007
                                 + 0.667387 w_k1_AXP_s007
                                 + 0.782898 w_k1_DIS_s007
                                 + 0.752865 w_k1_FDX_s007
                                 + 0.89612 w_k1_IBM_s007
                                 + 0.752391 w_k1_INTC_s007
                                 + 0.92386 w_k1_KO_s007
                                 + 0.743284 w_k1_MAT_s007
                                 + 0.852217 w_k1_MSFT_s007
                                 + 0.902388 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.957078 w_k1_ADM_s007
                                 + 1.44528 w_k1_AXP_s007
                                 + 1.39661 w_k1_DIS_s007
                                 + 1.30351 w_k1_FDX_s007 + 1.2572 w_k1_IBM_s007
                                 + 1.09106 w_k1_INTC_s007
                                 + 1.14073 w_k1_KO_s007 + 1.22532 w_k1_MAT_s007
                                 + 1.11871 w_k1_MSFT_s007 + 1.1647 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.50281 w_k1_ADM_s007
                                 + 1.31643 w_k1_AXP_s007
                                 + 1.02251 w_k1_DIS_s007
                                 + 1.23445 w_k1_FDX_s007
                                 + 1.01775 w_k1_IBM_s007
                                 + 0.91086 w_k1_INTC_s007
                                 + 0.951067 w_k1_KO_s007
                                 + 0.82897 w_k1_MAT_s007
                                 + 0.846911 w_k1_MSFT_s007
                                 + 1.10212 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.924909 w_k1_ADM_s007
                                 + 1.02 w_k1_AXP_s007 + 1.19774 w_k1_DIS_s007
                                 + 1.04017 w_k1_FDX_s007
                                 + 1.04546 w_k1_IBM_s007
                                 + 1.04204 w_k1_INTC_s007
                                 + 1.08059 w_k1_KO_s007
                                 + 0.595508 w_k1_MAT_s007
                                 + 0.989127 w_k1_MSFT_s007
                                 + 0.965832 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.13182 w_k1_ADM_s007
                                 + 1.13733 w_k1_AXP_s007
                                 + 1.47128 w_k1_DIS_s007
                                 + 1.38845 w_k1_FDX_s007
                                 + 1.00492 w_k1_IBM_s007
                                 + 1.11307 w_k1_INTC_s007
                                 + 1.09504 w_k1_KO_s007
                                 + 0.779054 w_k1_MAT_s007
                                 + 1.08763 w_k1_MSFT_s007
                                 + 1.18863 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.801888 w_k1_ADM_s007
                                 + 0.857519 w_k1_AXP_s007
                                 + 0.708324 w_k1_DIS_s007
                                 + 0.803134 w_k1_FDX_s007
                                 + 0.778066 w_k1_IBM_s007
                                 + 0.718237 w_k1_INTC_s007
                                 + 0.827321 w_k1_KO_s007
                                 + 0.682336 w_k1_MAT_s007
                                 + 0.734866 w_k1_MSFT_s007
                                 + 0.849624 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.951393 w_k1_ADM_s007
                                 + 1.12629 w_k1_AXP_s007
                                 + 1.02347 w_k1_DIS_s007
                                 + 0.959847 w_k1_FDX_s007
                                 + 1.46803 w_k1_IBM_s007
                                 + 0.989562 w_k1_INTC_s007
                                 + 1.12569 w_k1_KO_s007 + 1.05228 w_k1_MAT_s007
                                 + 0.976183 w_k1_MSFT_s007
                                 + 0.897282 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.02137 w_k1_ADM_s007
                                 + 0.990163 w_k1_AXP_s007
                                 + 0.823695 w_k1_DIS_s007
                                 + 1.00167 w_k1_FDX_s007
                                 + 0.780872 w_k1_IBM_s007
                                 + 0.78633 w_k1_INTC_s007
                                 + 0.734753 w_k1_KO_s007
                                 + 0.699666 w_k1_MAT_s007
                                 + 0.904562 w_k1_MSFT_s007
                                 + 0.86102 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.738381 w_k1_ADM_s010
                                 + 0.773093 w_k1_AXP_s010
                                 + 0.804232 w_k1_DIS_s010
                                 + 0.751888 w_k1_FDX_s010
                                 + 1.1246 w_k1_IBM_s010
                                 + 0.959682 w_k1_INTC_s010
                                 + 0.984931 w_k1_KO_s010
                                 + 0.963308 w_k1_MAT_s010
                                 + 0.903285 w_k1_MSFT_s010
                                 + 1.02096 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.14444 w_k1_ADM_s010
                                 + 1.39485 w_k1_AXP_s010
                                 + 1.14841 w_k1_DIS_s010
                                 + 1.16888 w_k1_FDX_s010
                                 + 0.924098 w_k1_IBM_s010
                                 + 1.20301 w_k1_INTC_s010
                                 + 0.894188 w_k1_KO_s010
                                 + 0.930318 w_k1_MAT_s010
                                 + 1.15552 w_k1_MSFT_s010
                                 + 1.15108 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.0685 w_k1_ADM_s010
                                 + 1.12638 w_k1_AXP_s010
                                 + 1.13271 w_k1_DIS_s010
                                 + 1.03572 w_k1_FDX_s010
                                 + 1.42381 w_k1_IBM_s010
                                 + 1.10405 w_k1_INTC_s010
                                 + 1.11537 w_k1_KO_s010 + 1.18149 w_k1_MAT_s010
                                 + 1.08536 w_k1_MSFT_s010
                                 + 1.14016 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10792 w_k1_ADM_s010
                                 + 1.36912 w_k1_AXP_s010 + 1.7021 w_k1_DIS_s010
                                 + 1.24594 w_k1_FDX_s010
                                 + 1.63521 w_k1_IBM_s010
                                 + 1.02748 w_k1_INTC_s010
                                 + 1.47157 w_k1_KO_s010 + 2.1122 w_k1_MAT_s010
                                 + 1.21153 w_k1_MSFT_s010
                                 + 1.20281 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.22749 w_k1_ADM_s010
                                 + 1.19771 w_k1_AXP_s010
                                 + 1.61175 w_k1_DIS_s010
                                 + 1.27643 w_k1_FDX_s010
                                 + 1.04177 w_k1_IBM_s010
                                 + 1.13793 w_k1_INTC_s010
                                 + 1.40527 w_k1_KO_s010 + 1.58004 w_k1_MAT_s010
                                 + 1.35184 w_k1_MSFT_s010
                                 + 1.24289 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.942008 w_k1_ADM_s010
                                 + 0.879251 w_k1_AXP_s010
                                 + 0.981454 w_k1_DIS_s010
                                 + 1.00033 w_k1_FDX_s010
                                 + 0.841673 w_k1_IBM_s010
                                 + 0.957554 w_k1_INTC_s010
                                 + 0.928061 w_k1_KO_s010
                                 + 1.11869 w_k1_MAT_s010
                                 + 1.01566 w_k1_MSFT_s010
                                 + 1.02371 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.782648 w_k1_ADM_s010
                                 + 0.876559 w_k1_AXP_s010
                                 + 0.790418 w_k1_DIS_s010
                                 + 0.84778 w_k1_FDX_s010
                                 + 0.700983 w_k1_IBM_s010
                                 + 0.922735 w_k1_INTC_s010
                                 + 0.775402 w_k1_KO_s010
                                 + 0.968345 w_k1_MAT_s010
                                 + 0.814696 w_k1_MSFT_s010
                                 + 0.899827 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.88057 w_k1_ADM_s010
                                 + 0.947431 w_k1_AXP_s010
                                 + 1.2519 w_k1_DIS_s010 + 1.04794 w_k1_FDX_s010
                                 + 1.03163 w_k1_IBM_s010
                                 + 1.07215 w_k1_INTC_s010
                                 + 1.03175 w_k1_KO_s010 + 1.1971 w_k1_MAT_s010
                                 + 1.10751 w_k1_MSFT_s010
                                 + 1.13754 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.94439 w_k1_ADM_s011
                                 + 1.03359 w_k1_AXP_s011
                                 + 1.09796 w_k1_DIS_s011
                                 + 0.953679 w_k1_FDX_s011
                                 + 1.13536 w_k1_IBM_s011
                                 + 1.19528 w_k1_INTC_s011
                                 + 1.14704 w_k1_KO_s011 + 1.01753 w_k1_MAT_s011
                                 + 0.960888 w_k1_MSFT_s011
                                 + 1.03863 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.83361 w_k1_ADM_s011
                                 + 0.82748 w_k1_AXP_s011
                                 + 0.783741 w_k1_DIS_s011
                                 + 0.815621 w_k1_FDX_s011
                                 + 0.974511 w_k1_IBM_s011
                                 + 0.938985 w_k1_INTC_s011
                                 + 1.07327 w_k1_KO_s011
                                 + 0.708487 w_k1_MAT_s011
                                 + 0.888983 w_k1_MSFT_s011
                                 + 1.0408 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.52758 w_k1_ADM_s011
                                 + 1.59824 w_k1_AXP_s011
                                 + 1.69151 w_k1_DIS_s011
                                 + 1.44086 w_k1_FDX_s011
                                 + 1.07821 w_k1_IBM_s011
                                 + 1.29492 w_k1_INTC_s011 + 1.2107 w_k1_KO_s011
                                 + 1.32182 w_k1_MAT_s011
                                 + 1.19314 w_k1_MSFT_s011
                                 + 1.30096 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.944994 w_k1_ADM_s011
                                 + 1.07588 w_k1_AXP_s011
                                 + 0.942869 w_k1_DIS_s011
                                 + 0.940976 w_k1_FDX_s011
                                 + 1.48034 w_k1_IBM_s011
                                 + 0.945671 w_k1_INTC_s011
                                 + 1.32646 w_k1_KO_s011 + 1.3414 w_k1_MAT_s011
                                 + 0.989614 w_k1_MSFT_s011
                                 + 1.08788 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.939471 w_k1_ADM_s011
                                 + 0.655776 w_k1_AXP_s011
                                 + 0.743413 w_k1_DIS_s011
                                 + 0.920373 w_k1_FDX_s011
                                 + 0.692143 w_k1_IBM_s011
                                 + 1.02925 w_k1_INTC_s011
                                 + 0.856262 w_k1_KO_s011
                                 + 0.518065 w_k1_MAT_s011
                                 + 0.933116 w_k1_MSFT_s011
                                 + 0.964018 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.22198 w_k1_ADM_s011
                                 + 0.971315 w_k1_AXP_s011
                                 + 1.38275 w_k1_DIS_s011
                                 + 1.20069 w_k1_FDX_s011
                                 + 0.810588 w_k1_IBM_s011
                                 + 1.12433 w_k1_INTC_s011
                                 + 1.14273 w_k1_KO_s011
                                 + 0.722495 w_k1_MAT_s011
                                 + 1.31699 w_k1_MSFT_s011
                                 + 1.12726 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.07031 w_k1_ADM_s011
                                 + 1.31812 w_k1_AXP_s011
                                 + 1.50256 w_k1_DIS_s011
                                 + 1.15245 w_k1_FDX_s011
                                 + 1.26769 w_k1_IBM_s011
                                 + 1.36959 w_k1_INTC_s011
                                 + 1.42523 w_k1_KO_s011 + 1.18237 w_k1_MAT_s011
                                 + 1.20616 w_k1_MSFT_s011
                                 + 1.31469 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.022 w_k1_ADM_s011
                                 + 1.61296 w_k1_AXP_s011
                                 + 1.52133 w_k1_DIS_s011
                                 + 1.18201 w_k1_FDX_s011
                                 + 1.84245 w_k1_IBM_s011
                                 + 1.33859 w_k1_INTC_s011
                                 + 1.57003 w_k1_KO_s011 + 2.01841 w_k1_MAT_s011
                                 + 1.34224 w_k1_MSFT_s011
                                 + 1.30994 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.94204 w_k1_ADM_s012
                                 + 0.776537 w_k1_AXP_s012
                                 + 0.871848 w_k1_DIS_s012
                                 + 0.810315 w_k1_FDX_s012
                                 + 0.927071 w_k1_IBM_s012
                                 + 0.96248 w_k1_INTC_s012
                                 + 0.917414 w_k1_KO_s012
                                 + 0.826573 w_k1_MAT_s012
                                 + 0.991467 w_k1_MSFT_s012
                                 + 0.951372 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.789783 w_k1_ADM_s012
                                 + 0.806521 w_k1_AXP_s012
                                 + 0.587745 w_k1_DIS_s012
                                 + 0.766213 w_k1_FDX_s012
                                 + 0.69142 w_k1_IBM_s012
                                 + 0.815804 w_k1_INTC_s012
                                 + 0.736541 w_k1_KO_s012
                                 + 0.735151 w_k1_MAT_s012
                                 + 0.788044 w_k1_MSFT_s012
                                 + 0.763495 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.15756 w_k1_ADM_s012
                                 + 1.30416 w_k1_AXP_s012
                                 + 1.24403 w_k1_DIS_s012
                                 + 1.19193 w_k1_FDX_s012
                                 + 0.910599 w_k1_IBM_s012
                                 + 1.20394 w_k1_INTC_s012
                                 + 1.07207 w_k1_KO_s012 + 1.00084 w_k1_MAT_s012
                                 + 1.06969 w_k1_MSFT_s012
                                 + 1.05109 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.08621 w_k1_ADM_s012
                                 + 1.09801 w_k1_AXP_s012
                                 + 1.33808 w_k1_DIS_s012
                                 + 1.01951 w_k1_FDX_s012
                                 + 1.27097 w_k1_IBM_s012
                                 + 1.32076 w_k1_INTC_s012
                                 + 1.25562 w_k1_KO_s012
                                 + 0.556075 w_k1_MAT_s012
                                 + 1.12239 w_k1_MSFT_s012
                                 + 1.03962 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.819855 w_k1_ADM_s012
                                 + 0.853771 w_k1_AXP_s012
                                 + 1.03542 w_k1_DIS_s012
                                 + 0.974476 w_k1_FDX_s012
                                 + 0.710209 w_k1_IBM_s012
                                 + 1.07442 w_k1_INTC_s012
                                 + 0.947511 w_k1_KO_s012
                                 + 0.589336 w_k1_MAT_s012
                                 + 1.09459 w_k1_MSFT_s012
                                 + 0.848774 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.02777 w_k1_ADM_s012
                                 + 1.17667 w_k1_AXP_s012
                                 + 0.820781 w_k1_DIS_s012
                                 + 0.824371 w_k1_FDX_s012
                                 + 1.62816 w_k1_IBM_s012
                                 + 0.964696 w_k1_INTC_s012
                                 + 1.03461 w_k1_KO_s012 + 1.39667 w_k1_MAT_s012
                                 + 0.830586 w_k1_MSFT_s012
                                 + 0.856558 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.952644 w_k1_ADM_s012
                                 + 1.1516 w_k1_AXP_s012 + 1.02158 w_k1_DIS_s012
                                 + 0.943244 w_k1_FDX_s012
                                 + 1.2015 w_k1_IBM_s012
                                 + 1.02779 w_k1_INTC_s012 + 1.0803 w_k1_KO_s012
                                 + 1.4505 w_k1_MAT_s012
                                 + 0.985737 w_k1_MSFT_s012
                                 + 0.953318 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04929 w_k1_ADM_s012
                                 + 1.06054 w_k1_AXP_s012
                                 + 0.732219 w_k1_DIS_s012
                                 + 0.875926 w_k1_FDX_s012
                                 + 0.950962 w_k1_IBM_s012
                                 + 0.917382 w_k1_INTC_s012
                                 + 0.854966 w_k1_KO_s012
                                 + 0.977094 w_k1_MAT_s012
                                 + 0.772827 w_k1_MSFT_s012
                                 + 0.851101 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.914424 w_k1_ADM_s013
                                 + 0.935237 w_k1_AXP_s013
                                 + 0.747841 w_k1_DIS_s013
                                 + 0.80104 w_k1_FDX_s013
                                 + 0.975316 w_k1_IBM_s013
                                 + 0.721322 w_k1_INTC_s013
                                 + 0.896252 w_k1_KO_s013
                                 + 0.98679 w_k1_MAT_s013
                                 + 0.763011 w_k1_MSFT_s013
                                 + 0.839192 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.05396 w_k1_ADM_s013
                                 + 1.0337 w_k1_AXP_s013 + 1.03523 w_k1_DIS_s013
                                 + 1.02533 w_k1_FDX_s013
                                 + 1.08235 w_k1_IBM_s013
                                 + 0.944552 w_k1_INTC_s013
                                 + 0.979682 w_k1_KO_s013 + 1.1634 w_k1_MAT_s013
                                 + 1.02125 w_k1_MSFT_s013 + 1.03 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.11847 w_k1_ADM_s013
                                 + 1.01195 w_k1_AXP_s013
                                 + 1.08273 w_k1_DIS_s013
                                 + 1.04299 w_k1_FDX_s013
                                 + 0.807595 w_k1_IBM_s013
                                 + 0.902423 w_k1_INTC_s013
                                 + 0.928474 w_k1_KO_s013
                                 + 0.881367 w_k1_MAT_s013
                                 + 1.06358 w_k1_MSFT_s013
                                 + 0.95822 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.842232 w_k1_ADM_s013
                                 + 0.693296 w_k1_AXP_s013
                                 + 0.720714 w_k1_DIS_s013
                                 + 0.852603 w_k1_FDX_s013
                                 + 0.673958 w_k1_IBM_s013
                                 + 0.824911 w_k1_INTC_s013
                                 + 0.75579 w_k1_KO_s013
                                 + 0.498815 w_k1_MAT_s013
                                 + 0.713217 w_k1_MSFT_s013
                                 + 0.842778 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.56851 w_k1_ADM_s013
                                 + 1.48986 w_k1_AXP_s013
                                 + 1.62615 w_k1_DIS_s013 + 1.22 w_k1_FDX_s013
                                 + 1.58151 w_k1_IBM_s013
                                 + 1.08739 w_k1_INTC_s013
                                 + 1.29207 w_k1_KO_s013 + 1.46332 w_k1_MAT_s013
                                 + 1.05698 w_k1_MSFT_s013 + 1.1343 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.81541 w_k1_ADM_s013
                                 + 1.04098 w_k1_AXP_s013
                                 + 0.849674 w_k1_DIS_s013
                                 + 0.995579 w_k1_FDX_s013
                                 + 0.766748 w_k1_IBM_s013
                                 + 1.02226 w_k1_INTC_s013
                                 + 0.851204 w_k1_KO_s013
                                 + 0.632219 w_k1_MAT_s013
                                 + 0.890113 w_k1_MSFT_s013
                                 + 0.883471 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.1988 w_k1_ADM_s013
                                 + 1.40818 w_k1_AXP_s013
                                 + 1.24076 w_k1_DIS_s013
                                 + 1.12875 w_k1_FDX_s013
                                 + 1.10079 w_k1_IBM_s013
                                 + 1.12804 w_k1_INTC_s013
                                 + 1.17774 w_k1_KO_s013 + 1.27965 w_k1_MAT_s013
                                 + 0.990675 w_k1_MSFT_s013
                                 + 1.10188 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.972388 w_k1_ADM_s013
                                 + 1.07935 w_k1_AXP_s013
                                 + 0.975954 w_k1_DIS_s013
                                 + 0.885546 w_k1_FDX_s013
                                 + 1.47451 w_k1_IBM_s013
                                 + 0.912167 w_k1_INTC_s013
                                 + 1.09443 w_k1_KO_s013 + 1.36304 w_k1_MAT_s013
                                 + 0.885029 w_k1_MSFT_s013 + 1.019 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.969082 w_k1_ADM_s014
                                 + 1.19367 w_k1_AXP_s014
                                 + 1.07014 w_k1_DIS_s014
                                 + 0.959397 w_k1_FDX_s014
                                 + 1.27668 w_k1_IBM_s014
                                 + 1.09701 w_k1_INTC_s014
                                 + 0.937948 w_k1_KO_s014
                                 + 0.753603 w_k1_MAT_s014
                                 + 0.884006 w_k1_MSFT_s014
                                 + 1.05732 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.936771 w_k1_ADM_s014
                                 + 0.954745 w_k1_AXP_s014
                                 + 0.96197 w_k1_DIS_s014
                                 + 0.901063 w_k1_FDX_s014
                                 + 0.95593 w_k1_IBM_s014
                                 + 1.01295 w_k1_INTC_s014
                                 + 0.896135 w_k1_KO_s014
                                 + 0.768191 w_k1_MAT_s014
                                 + 0.965546 w_k1_MSFT_s014
                                 + 0.923693 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.95141 w_k1_ADM_s014
                                 + 0.954021 w_k1_AXP_s014
                                 + 0.983534 w_k1_DIS_s014
                                 + 0.835717 w_k1_FDX_s014
                                 + 1.51072 w_k1_IBM_s014
                                 + 0.949612 w_k1_INTC_s014
                                 + 1.07607 w_k1_KO_s014
                                 + 0.976993 w_k1_MAT_s014
                                 + 0.922963 w_k1_MSFT_s014
                                 + 1.08254 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.3914 w_k1_ADM_s014
                                 + 1.63806 w_k1_AXP_s014
                                 + 1.63307 w_k1_DIS_s014
                                 + 1.27108 w_k1_FDX_s014 + 1.4754 w_k1_IBM_s014
                                 + 1.31567 w_k1_INTC_s014
                                 + 1.38931 w_k1_KO_s014 + 1.63074 w_k1_MAT_s014
                                 + 1.11893 w_k1_MSFT_s014
                                 + 1.29044 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.34209 w_k1_ADM_s014
                                 + 1.72426 w_k1_AXP_s014
                                 + 1.95213 w_k1_DIS_s014
                                 + 1.49078 w_k1_FDX_s014
                                 + 0.927543 w_k1_IBM_s014
                                 + 1.30238 w_k1_INTC_s014 + 1.3092 w_k1_KO_s014
                                 + 1.00835 w_k1_MAT_s014
                                 + 1.35986 w_k1_MSFT_s014
                                 + 1.35441 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.27775 w_k1_ADM_s014
                                 + 1.09138 w_k1_AXP_s014
                                 + 1.13291 w_k1_DIS_s014
                                 + 1.16937 w_k1_FDX_s014
                                 + 0.834698 w_k1_IBM_s014
                                 + 0.940871 w_k1_INTC_s014
                                 + 0.878752 w_k1_KO_s014
                                 + 0.692222 w_k1_MAT_s014
                                 + 0.962137 w_k1_MSFT_s014
                                 + 1.06289 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.12011 w_k1_ADM_s014
                                 + 0.825837 w_k1_AXP_s014
                                 + 0.800372 w_k1_DIS_s014
                                 + 0.882833 w_k1_FDX_s014
                                 + 0.588009 w_k1_IBM_s014
                                 + 0.894045 w_k1_INTC_s014
                                 + 0.5709 w_k1_KO_s014 + 0.362691 w_k1_MAT_s014
                                 + 0.856926 w_k1_MSFT_s014
                                 + 0.853827 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.875065 w_k1_ADM_s014
                                 + 0.840258 w_k1_AXP_s014
                                 + 0.665804 w_k1_DIS_s014
                                 + 0.736301 w_k1_FDX_s014
                                 + 0.820913 w_k1_IBM_s014
                                 + 0.874885 w_k1_INTC_s014
                                 + 0.653143 w_k1_KO_s014
                                 + 0.531289 w_k1_MAT_s014
                                 + 0.751254 w_k1_MSFT_s014
                                 + 0.825785 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.741621 w_k1_ADM_s015
                                 + 0.700117 w_k1_AXP_s015
                                 + 0.701445 w_k1_DIS_s015
                                 + 0.73992 w_k1_FDX_s015
                                 + 0.797722 w_k1_IBM_s015
                                 + 0.718049 w_k1_INTC_s015
                                 + 0.82932 w_k1_KO_s015
                                 + 0.865316 w_k1_MAT_s015
                                 + 0.802709 w_k1_MSFT_s015
                                 + 0.768277 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.17814 w_k1_ADM_s015
                                 + 1.40788 w_k1_AXP_s015
                                 + 1.67844 w_k1_DIS_s015
                                 + 1.41461 w_k1_FDX_s015
                                 + 1.37326 w_k1_IBM_s015
                                 + 1.16105 w_k1_INTC_s015
                                 + 1.36416 w_k1_KO_s015 + 1.47015 w_k1_MAT_s015
                                 + 1.38078 w_k1_MSFT_s015 + 1.1217 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.907004 w_k1_ADM_s015
                                 + 0.972497 w_k1_AXP_s015
                                 + 0.881825 w_k1_DIS_s015
                                 + 0.853481 w_k1_FDX_s015 + 1.642 w_k1_IBM_s015
                                 + 0.907543 w_k1_INTC_s015
                                 + 1.13791 w_k1_KO_s015 + 1.29718 w_k1_MAT_s015
                                 + 0.892954 w_k1_MSFT_s015
                                 + 0.982503 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.878346 w_k1_ADM_s015
                                 + 0.760897 w_k1_AXP_s015
                                 + 0.920373 w_k1_DIS_s015
                                 + 0.890064 w_k1_FDX_s015
                                 + 1.13168 w_k1_IBM_s015
                                 + 0.868886 w_k1_INTC_s015
                                 + 0.963443 w_k1_KO_s015
                                 + 0.923448 w_k1_MAT_s015
                                 + 1.0151 w_k1_MSFT_s015
                                 + 0.965358 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.26234 w_k1_ADM_s015
                                 + 1.26575 w_k1_AXP_s015
                                 + 1.22062 w_k1_DIS_s015
                                 + 1.19025 w_k1_FDX_s015
                                 + 1.21006 w_k1_IBM_s015
                                 + 1.12925 w_k1_INTC_s015
                                 + 1.09424 w_k1_KO_s015 + 1.25092 w_k1_MAT_s015
                                 + 1.16262 w_k1_MSFT_s015
                                 + 1.14439 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.09245 w_k1_ADM_s015
                                 + 0.751742 w_k1_AXP_s015
                                 + 0.885259 w_k1_DIS_s015
                                 + 1.06801 w_k1_FDX_s015
                                 + 0.64297 w_k1_IBM_s015
                                 + 0.88334 w_k1_INTC_s015
                                 + 0.811258 w_k1_KO_s015
                                 + 0.578568 w_k1_MAT_s015
                                 + 1.04373 w_k1_MSFT_s015
                                 + 0.862954 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.22885 w_k1_ADM_s015
                                 + 1.11119 w_k1_AXP_s015
                                 + 1.01036 w_k1_DIS_s015
                                 + 1.06184 w_k1_FDX_s015
                                 + 0.892006 w_k1_IBM_s015
                                 + 0.893183 w_k1_INTC_s015
                                 + 0.891682 w_k1_KO_s015
                                 + 0.866873 w_k1_MAT_s015
                                 + 1.03406 w_k1_MSFT_s015
                                 + 0.971416 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.08251 w_k1_ADM_s015
                                 + 0.847767 w_k1_AXP_s015
                                 + 1.47609 w_k1_DIS_s015
                                 + 1.12274 w_k1_FDX_s015
                                 + 1.04805 w_k1_IBM_s015
                                 + 1.03959 w_k1_INTC_s015
                                 + 1.14935 w_k1_KO_s015
                                 + 0.952515 w_k1_MAT_s015
                                 + 1.26962 w_k1_MSFT_s015
                                 + 0.997622 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.29176 w_k1_ADM_s016
                                 + 1.42132 w_k1_AXP_s016 + 1.4432 w_k1_DIS_s016
                                 + 1.39571 w_k1_FDX_s016
                                 + 0.931627 w_k1_IBM_s016
                                 + 1.13849 w_k1_INTC_s016
                                 + 1.14929 w_k1_KO_s016 + 1.1091 w_k1_MAT_s016
                                 + 1.3463 w_k1_MSFT_s016 + 1.02972 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.947569 w_k1_ADM_s016
                                 + 1.35383 w_k1_AXP_s016
                                 + 1.69697 w_k1_DIS_s016 + 1.2428 w_k1_FDX_s016
                                 + 1.52 w_k1_IBM_s016 + 0.982774 w_k1_INTC_s016
                                 + 1.61062 w_k1_KO_s016 + 1.8025 w_k1_MAT_s016
                                 + 1.21465 w_k1_MSFT_s016
                                 + 1.17263 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.677 w_k1_ADM_s016
                                 + 1.0224 w_k1_AXP_s016 + 1.0136 w_k1_DIS_s016
                                 + 0.825411 w_k1_FDX_s016
                                 + 1.5756 w_k1_IBM_s016
                                 + 1.05325 w_k1_INTC_s016
                                 + 1.36755 w_k1_KO_s016 + 1.1817 w_k1_MAT_s016
                                 + 0.821749 w_k1_MSFT_s016
                                 + 1.00795 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.946929 w_k1_ADM_s016
                                 + 0.937281 w_k1_AXP_s016
                                 + 1.34645 w_k1_DIS_s016
                                 + 1.07722 w_k1_FDX_s016
                                 + 0.992697 w_k1_IBM_s016
                                 + 1.04287 w_k1_INTC_s016
                                 + 1.23932 w_k1_KO_s016
                                 + 0.981235 w_k1_MAT_s016
                                 + 1.16515 w_k1_MSFT_s016 + 1.076 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.846156 w_k1_ADM_s016
                                 + 1.34368 w_k1_AXP_s016
                                 + 1.08521 w_k1_DIS_s016
                                 + 1.03685 w_k1_FDX_s016
                                 + 1.30518 w_k1_IBM_s016
                                 + 0.849126 w_k1_INTC_s016
                                 + 1.14226 w_k1_KO_s016 + 1.37989 w_k1_MAT_s016
                                 + 0.789514 w_k1_MSFT_s016
                                 + 0.983112 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.04351 w_k1_ADM_s016
                                 + 1.20114 w_k1_AXP_s016
                                 + 1.03436 w_k1_DIS_s016
                                 + 1.20127 w_k1_FDX_s016
                                 + 0.64399 w_k1_IBM_s016
                                 + 1.01219 w_k1_INTC_s016
                                 + 0.928053 w_k1_KO_s016
                                 + 0.716214 w_k1_MAT_s016
                                 + 1.14858 w_k1_MSFT_s016
                                 + 1.03091 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.8513 w_k1_ADM_s016
                                 + 0.879853 w_k1_AXP_s016
                                 + 0.822689 w_k1_DIS_s016
                                 + 0.83701 w_k1_FDX_s016
                                 + 0.848976 w_k1_IBM_s016
                                 + 0.906663 w_k1_INTC_s016
                                 + 0.919257 w_k1_KO_s016
                                 + 0.550447 w_k1_MAT_s016
                                 + 0.779065 w_k1_MSFT_s016
                                 + 0.860576 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.850622 w_k1_ADM_s016
                                 + 0.928089 w_k1_AXP_s016
                                 + 0.94747 w_k1_DIS_s016
                                 + 0.97942 w_k1_FDX_s016
                                 + 1.07854 w_k1_IBM_s016
                                 + 0.858319 w_k1_INTC_s016
                                 + 0.950781 w_k1_KO_s016
                                 + 1.03537 w_k1_MAT_s016
                                 + 0.90316 w_k1_MSFT_s016
                                 + 0.940382 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.27079 w_k1_ADM_s017
                                 + 0.865864 w_k1_AXP_s017
                                 + 0.909536 w_k1_DIS_s017
                                 + 1.07266 w_k1_FDX_s017
                                 + 0.789988 w_k1_IBM_s017
                                 + 1.12145 w_k1_INTC_s017
                                 + 0.944238 w_k1_KO_s017
                                 + 0.745861 w_k1_MAT_s017
                                 + 1.09731 w_k1_MSFT_s017
                                 + 0.950134 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.33922 w_k1_ADM_s017
                                 + 1.26586 w_k1_AXP_s017
                                 + 1.40268 w_k1_DIS_s017
                                 + 1.39176 w_k1_FDX_s017
                                 + 0.991108 w_k1_IBM_s017
                                 + 1.22382 w_k1_INTC_s017
                                 + 1.16958 w_k1_KO_s017
                                 + 0.948209 w_k1_MAT_s017
                                 + 1.13775 w_k1_MSFT_s017
                                 + 1.04395 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.09497 w_k1_ADM_s017
                                 + 0.71136 w_k1_AXP_s017
                                 + 0.805533 w_k1_DIS_s017
                                 + 0.931373 w_k1_FDX_s017
                                 + 0.595146 w_k1_IBM_s017
                                 + 0.760223 w_k1_INTC_s017
                                 + 0.79037 w_k1_KO_s017
                                 + 0.464081 w_k1_MAT_s017
                                 + 0.852902 w_k1_MSFT_s017
                                 + 0.817348 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.04321 w_k1_ADM_s017
                                 + 0.973388 w_k1_AXP_s017
                                 + 0.824275 w_k1_DIS_s017
                                 + 0.93035 w_k1_FDX_s017
                                 + 1.64145 w_k1_IBM_s017
                                 + 0.843806 w_k1_INTC_s017
                                 + 1.03742 w_k1_KO_s017 + 1.90699 w_k1_MAT_s017
                                 + 0.912964 w_k1_MSFT_s017
                                 + 1.04175 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.15668 w_k1_ADM_s017
                                 + 0.929917 w_k1_AXP_s017
                                 + 1.10571 w_k1_DIS_s017
                                 + 1.07527 w_k1_FDX_s017
                                 + 1.15333 w_k1_IBM_s017
                                 + 1.01812 w_k1_INTC_s017
                                 + 1.20349 w_k1_KO_s017 + 1.04133 w_k1_MAT_s017
                                 + 1.03987 w_k1_MSFT_s017
                                 + 1.10614 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.827565 w_k1_ADM_s017
                                 + 0.676021 w_k1_AXP_s017
                                 + 0.630513 w_k1_DIS_s017
                                 + 0.725057 w_k1_FDX_s017
                                 + 0.977519 w_k1_IBM_s017
                                 + 0.802012 w_k1_INTC_s017
                                 + 0.805747 w_k1_KO_s017
                                 + 0.543681 w_k1_MAT_s017
                                 + 0.778157 w_k1_MSFT_s017
                                 + 0.883443 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.04574 w_k1_ADM_s017
                                 + 0.68785 w_k1_AXP_s017
                                 + 0.80702 w_k1_DIS_s017
                                 + 0.838354 w_k1_FDX_s017
                                 + 1.08963 w_k1_IBM_s017
                                 + 0.783343 w_k1_INTC_s017
                                 + 1.02174 w_k1_KO_s017
                                 + 0.879227 w_k1_MAT_s017
                                 + 0.86321 w_k1_MSFT_s017
                                 + 1.02194 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.10428 w_k1_ADM_s017
                                 + 1.32332 w_k1_AXP_s017
                                 + 1.24515 w_k1_DIS_s017
                                 + 1.12293 w_k1_FDX_s017
                                 + 2.11144 w_k1_IBM_s017
                                 + 0.79336 w_k1_INTC_s017
                                 + 1.52033 w_k1_KO_s017 + 2.38329 w_k1_MAT_s017
                                 + 1.00442 w_k1_MSFT_s017
                                 + 1.16286 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.37991 w_k1_ADM_s100
                                 + 1.11158 w_k1_AXP_s100
                                 + 1.13566 w_k1_DIS_s100
                                 + 1.17307 w_k1_FDX_s100
                                 + 0.963154 w_k1_IBM_s100
                                 + 1.13007 w_k1_INTC_s100 + 1.1101 w_k1_KO_s100
                                 + 0.841504 w_k1_MAT_s100
                                 + 1.11838 w_k1_MSFT_s100
                                 + 1.22722 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.732938 w_k1_ADM_s100
                                 + 0.596594 w_k1_AXP_s100
                                 + 0.61505 w_k1_DIS_s100
                                 + 0.755678 w_k1_FDX_s100
                                 + 0.748116 w_k1_IBM_s100
                                 + 0.778077 w_k1_INTC_s100
                                 + 0.630633 w_k1_KO_s100
                                 + 0.450413 w_k1_MAT_s100
                                 + 0.705761 w_k1_MSFT_s100
                                 + 0.767314 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.739555 w_k1_ADM_s100
                                 + 0.888938 w_k1_AXP_s100
                                 + 0.959004 w_k1_DIS_s100
                                 + 0.8823 w_k1_FDX_s100 + 1.44645 w_k1_IBM_s100
                                 + 1.11101 w_k1_INTC_s100
                                 + 1.19277 w_k1_KO_s100 + 1.27823 w_k1_MAT_s100
                                 + 0.843932 w_k1_MSFT_s100
                                 + 1.08266 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.0025 w_k1_ADM_s100
                                 + 0.823953 w_k1_AXP_s100
                                 + 0.825791 w_k1_DIS_s100
                                 + 0.909665 w_k1_FDX_s100
                                 + 0.663314 w_k1_IBM_s100
                                 + 0.858259 w_k1_INTC_s100
                                 + 0.827344 w_k1_KO_s100
                                 + 0.488725 w_k1_MAT_s100
                                 + 0.925966 w_k1_MSFT_s100
                                 + 0.818063 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.19706 w_k1_ADM_s100
                                 + 0.868714 w_k1_AXP_s100
                                 + 0.90404 w_k1_DIS_s100
                                 + 0.894733 w_k1_FDX_s100
                                 + 0.967576 w_k1_IBM_s100
                                 + 0.769586 w_k1_INTC_s100
                                 + 0.837949 w_k1_KO_s100
                                 + 0.662511 w_k1_MAT_s100
                                 + 0.838416 w_k1_MSFT_s100
                                 + 1.02804 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.914774 w_k1_ADM_s100
                                 + 0.734498 w_k1_AXP_s100
                                 + 0.728888 w_k1_DIS_s100
                                 + 0.774184 w_k1_FDX_s100
                                 + 1.19451 w_k1_IBM_s100
                                 + 0.663141 w_k1_INTC_s100
                                 + 1.00739 w_k1_KO_s100 + 1.07703 w_k1_MAT_s100
                                 + 0.673819 w_k1_MSFT_s100
                                 + 0.883178 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.948683 w_k1_ADM_s100
                                 + 1.06305 w_k1_AXP_s100
                                 + 1.25389 w_k1_DIS_s100
                                 + 1.02201 w_k1_FDX_s100
                                 + 1.17988 w_k1_IBM_s100
                                 + 1.0483 w_k1_INTC_s100 + 1.20486 w_k1_KO_s100
                                 + 0.9275 w_k1_MAT_s100
                                 + 0.998811 w_k1_MSFT_s100
                                 + 1.01172 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.41202 w_k1_ADM_s100
                                 + 1.38467 w_k1_AXP_s100
                                 + 1.65546 w_k1_DIS_s100
                                 + 1.39771 w_k1_FDX_s100
                                 + 1.35563 w_k1_IBM_s100
                                 + 1.31118 w_k1_INTC_s100
                                 + 1.43794 w_k1_KO_s100 + 1.01188 w_k1_MAT_s100
                                 + 1.19584 w_k1_MSFT_s100
                                 + 1.27924 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.880044 w_k1_ADM_s101
                                 + 0.775976 w_k1_AXP_s101
                                 + 0.805294 w_k1_DIS_s101
                                 + 0.807563 w_k1_FDX_s101
                                 + 0.97974 w_k1_IBM_s101
                                 + 1.03313 w_k1_INTC_s101
                                 + 0.902258 w_k1_KO_s101
                                 + 0.673139 w_k1_MAT_s101
                                 + 0.890864 w_k1_MSFT_s101
                                 + 0.81633 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.55733 w_k1_ADM_s101
                                 + 1.451 w_k1_AXP_s101 + 1.8826 w_k1_DIS_s101
                                 + 1.49172 w_k1_FDX_s101 + 1.187 w_k1_IBM_s101
                                 + 1.48814 w_k1_INTC_s101 + 1.539 w_k1_KO_s101
                                 + 0.9347 w_k1_MAT_s101
                                 + 1.48694 w_k1_MSFT_s101
                                 + 1.22823 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.75718 w_k1_ADM_s101
                                 + 1.70569 w_k1_AXP_s101
                                 + 1.91674 w_k1_DIS_s101
                                 + 1.61814 w_k1_FDX_s101
                                 + 1.94919 w_k1_IBM_s101
                                 + 1.5678 w_k1_INTC_s101 + 1.55126 w_k1_KO_s101
                                 + 1.73843 w_k1_MAT_s101
                                 + 1.52513 w_k1_MSFT_s101 + 1.3089 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.0607 w_k1_ADM_s101
                                 + 1.21075 w_k1_AXP_s101
                                 + 0.962148 w_k1_DIS_s101
                                 + 1.02066 w_k1_FDX_s101
                                 + 1.08968 w_k1_IBM_s101
                                 + 0.98676 w_k1_INTC_s101 + 1.0166 w_k1_KO_s101
                                 + 1.00577 w_k1_MAT_s101
                                 + 0.965099 w_k1_MSFT_s101
                                 + 0.966217 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.867162 w_k1_ADM_s101
                                 + 0.639718 w_k1_AXP_s101
                                 + 0.51241 w_k1_DIS_s101
                                 + 0.696842 w_k1_FDX_s101
                                 + 0.616463 w_k1_IBM_s101
                                 + 0.814002 w_k1_INTC_s101
                                 + 0.601752 w_k1_KO_s101
                                 + 0.433196 w_k1_MAT_s101
                                 + 0.755558 w_k1_MSFT_s101
                                 + 0.651296 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.0766 w_k1_ADM_s101
                                 + 0.939386 w_k1_AXP_s101
                                 + 1.03089 w_k1_DIS_s101
                                 + 1.09962 w_k1_FDX_s101
                                 + 0.777877 w_k1_IBM_s101
                                 + 1.41595 w_k1_INTC_s101
                                 + 0.854499 w_k1_KO_s101
                                 + 0.572248 w_k1_MAT_s101
                                 + 1.04141 w_k1_MSFT_s101
                                 + 0.888983 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.1377 w_k1_ADM_s101
                                 + 1.3876 w_k1_AXP_s101 + 1.23244 w_k1_DIS_s101
                                 + 1.21173 w_k1_FDX_s101
                                 + 1.23785 w_k1_IBM_s101
                                 + 1.22064 w_k1_INTC_s101
                                 + 1.27059 w_k1_KO_s101 + 1.52351 w_k1_MAT_s101
                                 + 1.22268 w_k1_MSFT_s101
                                 + 1.14194 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.20438 w_k1_ADM_s101
                                 + 1.74517 w_k1_AXP_s101
                                 + 1.59659 w_k1_DIS_s101
                                 + 1.38064 w_k1_FDX_s101
                                 + 1.46686 w_k1_IBM_s101
                                 + 1.4747 w_k1_INTC_s101 + 1.53969 w_k1_KO_s101
                                 + 1.18313 w_k1_MAT_s101
                                 + 1.31155 w_k1_MSFT_s101
                                 + 1.13443 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.819408 w_k1_ADM_s102
                                 + 0.905434 w_k1_AXP_s102
                                 + 0.902718 w_k1_DIS_s102
                                 + 0.995423 w_k1_FDX_s102
                                 + 0.747733 w_k1_IBM_s102
                                 + 0.870702 w_k1_INTC_s102
                                 + 0.755344 w_k1_KO_s102
                                 + 1.02122 w_k1_MAT_s102
                                 + 0.987218 w_k1_MSFT_s102
                                 + 1.0045 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.72274 w_k1_ADM_s102
                                 + 0.65899 w_k1_AXP_s102
                                 + 0.682854 w_k1_DIS_s102
                                 + 0.771326 w_k1_FDX_s102
                                 + 0.792358 w_k1_IBM_s102
                                 + 0.649221 w_k1_INTC_s102
                                 + 0.730874 w_k1_KO_s102
                                 + 0.934283 w_k1_MAT_s102
                                 + 0.818655 w_k1_MSFT_s102
                                 + 0.841291 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.849317 w_k1_ADM_s102
                                 + 1.01608 w_k1_AXP_s102
                                 + 1.07805 w_k1_DIS_s102
                                 + 1.01184 w_k1_FDX_s102
                                 + 1.15322 w_k1_IBM_s102
                                 + 1.04775 w_k1_INTC_s102
                                 + 0.970918 w_k1_KO_s102
                                 + 1.22967 w_k1_MAT_s102
                                 + 1.05425 w_k1_MSFT_s102
                                 + 1.05562 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.734919 w_k1_ADM_s102
                                 + 0.862106 w_k1_AXP_s102
                                 + 0.96131 w_k1_DIS_s102
                                 + 0.870278 w_k1_FDX_s102
                                 + 1.41035 w_k1_IBM_s102
                                 + 0.933496 w_k1_INTC_s102
                                 + 1.06772 w_k1_KO_s102 + 1.13091 w_k1_MAT_s102
                                 + 0.93298 w_k1_MSFT_s102
                                 + 1.01726 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.890713 w_k1_ADM_s102
                                 + 1.34103 w_k1_AXP_s102
                                 + 1.34629 w_k1_DIS_s102 + 1.2136 w_k1_FDX_s102
                                 + 1.4525 w_k1_IBM_s102
                                 + 1.23718 w_k1_INTC_s102
                                 + 1.35551 w_k1_KO_s102 + 2.05006 w_k1_MAT_s102
                                 + 1.19036 w_k1_MSFT_s102
                                 + 1.07669 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.08756 w_k1_ADM_s102
                                 + 0.74444 w_k1_AXP_s102
                                 + 0.938769 w_k1_DIS_s102
                                 + 1.01934 w_k1_FDX_s102
                                 + 0.68757 w_k1_IBM_s102
                                 + 0.883185 w_k1_INTC_s102
                                 + 0.879784 w_k1_KO_s102
                                 + 0.649991 w_k1_MAT_s102
                                 + 0.954696 w_k1_MSFT_s102
                                 + 0.966735 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.15759 w_k1_ADM_s102
                                 + 1.21513 w_k1_AXP_s102
                                 + 1.28545 w_k1_DIS_s102
                                 + 1.28637 w_k1_FDX_s102
                                 + 0.922536 w_k1_IBM_s102
                                 + 0.979253 w_k1_INTC_s102
                                 + 0.957047 w_k1_KO_s102
                                 + 1.11167 w_k1_MAT_s102
                                 + 1.20257 w_k1_MSFT_s102
                                 + 1.12454 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.780081 w_k1_ADM_s102
                                 + 0.912706 w_k1_AXP_s102
                                 + 0.813288 w_k1_DIS_s102
                                 + 0.904624 w_k1_FDX_s102
                                 + 1.01891 w_k1_IBM_s102
                                 + 0.737839 w_k1_INTC_s102
                                 + 0.824816 w_k1_KO_s102
                                 + 1.21052 w_k1_MAT_s102
                                 + 0.845754 w_k1_MSFT_s102
                                 + 0.956122 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.893981 w_k1_ADM_s103
                                 + 1.46257 w_k1_AXP_s103
                                 + 1.58744 w_k1_DIS_s103
                                 + 1.27985 w_k1_FDX_s103 + 1.4064 w_k1_IBM_s103
                                 + 1.22429 w_k1_INTC_s103
                                 + 1.37632 w_k1_KO_s103 + 1.51094 w_k1_MAT_s103
                                 + 1.41326 w_k1_MSFT_s103
                                 + 1.13116 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.90057 w_k1_ADM_s103
                                 + 0.976284 w_k1_AXP_s103
                                 + 0.939393 w_k1_DIS_s103
                                 + 0.883366 w_k1_FDX_s103
                                 + 0.869446 w_k1_IBM_s103
                                 + 0.889142 w_k1_INTC_s103
                                 + 0.907274 w_k1_KO_s103
                                 + 0.889972 w_k1_MAT_s103
                                 + 0.923996 w_k1_MSFT_s103
                                 + 0.883004 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.86249 w_k1_ADM_s103
                                 + 1.12863 w_k1_AXP_s103
                                 + 0.81197 w_k1_DIS_s103
                                 + 0.979229 w_k1_FDX_s103
                                 + 1.0387 w_k1_IBM_s103
                                 + 0.884821 w_k1_INTC_s103
                                 + 0.939656 w_k1_KO_s103
                                 + 1.48334 w_k1_MAT_s103
                                 + 0.95346 w_k1_MSFT_s103
                                 + 1.10326 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.00378 w_k1_ADM_s103
                                 + 1.10415 w_k1_AXP_s103
                                 + 1.09033 w_k1_DIS_s103
                                 + 0.927448 w_k1_FDX_s103
                                 + 1.12784 w_k1_IBM_s103
                                 + 0.939898 w_k1_INTC_s103
                                 + 1.23612 w_k1_KO_s103 + 1.30654 w_k1_MAT_s103
                                 + 1.20357 w_k1_MSFT_s103 + 1.0112 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.962248 w_k1_ADM_s103
                                 + 1.24502 w_k1_AXP_s103
                                 + 1.19303 w_k1_DIS_s103
                                 + 1.13305 w_k1_FDX_s103
                                 + 0.771529 w_k1_IBM_s103
                                 + 1.19224 w_k1_INTC_s103
                                 + 1.01013 w_k1_KO_s103 + 1.02295 w_k1_MAT_s103
                                 + 1.26231 w_k1_MSFT_s103 + 1.0723 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.685899 w_k1_ADM_s103
                                 + 0.581132 w_k1_AXP_s103
                                 + 0.639983 w_k1_DIS_s103
                                 + 0.710217 w_k1_FDX_s103
                                 + 0.665865 w_k1_IBM_s103
                                 + 0.670795 w_k1_INTC_s103
                                 + 0.732339 w_k1_KO_s103
                                 + 0.519847 w_k1_MAT_s103
                                 + 0.823336 w_k1_MSFT_s103
                                 + 0.773656 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.843232 w_k1_ADM_s103
                                 + 0.920936 w_k1_AXP_s103
                                 + 0.768357 w_k1_DIS_s103
                                 + 0.929557 w_k1_FDX_s103
                                 + 0.697165 w_k1_IBM_s103
                                 + 0.93605 w_k1_INTC_s103
                                 + 0.768272 w_k1_KO_s103
                                 + 0.456915 w_k1_MAT_s103
                                 + 0.941112 w_k1_MSFT_s103
                                 + 0.930671 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.834271 w_k1_ADM_s103
                                 + 0.823749 w_k1_AXP_s103
                                 + 0.649766 w_k1_DIS_s103
                                 + 0.690571 w_k1_FDX_s103
                                 + 0.9929 w_k1_IBM_s103
                                 + 0.903486 w_k1_INTC_s103
                                 + 0.873489 w_k1_KO_s103
                                 + 0.954488 w_k1_MAT_s103
                                 + 0.878675 w_k1_MSFT_s103
                                 + 0.907046 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.903376 w_k1_ADM_s104
                                 + 0.876804 w_k1_AXP_s104
                                 + 0.794131 w_k1_DIS_s104
                                 + 0.830673 w_k1_FDX_s104
                                 + 1.01785 w_k1_IBM_s104
                                 + 1.02252 w_k1_INTC_s104
                                 + 0.91864 w_k1_KO_s104 + 1.14037 w_k1_MAT_s104
                                 + 0.910296 w_k1_MSFT_s104
                                 + 1.02712 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.05305 w_k1_ADM_s104
                                 + 1.00645 w_k1_AXP_s104
                                 + 1.24332 w_k1_DIS_s104 + 1.1791 w_k1_FDX_s104
                                 + 0.67535 w_k1_IBM_s104
                                 + 1.33981 w_k1_INTC_s104
                                 + 0.869007 w_k1_KO_s104
                                 + 0.392008 w_k1_MAT_s104
                                 + 1.08078 w_k1_MSFT_s104
                                 + 0.905691 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02598 w_k1_ADM_s104
                                 + 0.868188 w_k1_AXP_s104
                                 + 0.96619 w_k1_DIS_s104 + 1.0256 w_k1_FDX_s104
                                 + 0.8306 w_k1_IBM_s104
                                 + 1.30324 w_k1_INTC_s104
                                 + 0.947643 w_k1_KO_s104
                                 + 0.621997 w_k1_MAT_s104
                                 + 1.028 w_k1_MSFT_s104 + 1.01599 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.0328 w_k1_ADM_s104
                                 + 1.46161 w_k1_AXP_s104
                                 + 1.50339 w_k1_DIS_s104
                                 + 1.30911 w_k1_FDX_s104
                                 + 1.34581 w_k1_IBM_s104
                                 + 1.60025 w_k1_INTC_s104
                                 + 1.24768 w_k1_KO_s104 + 1.39132 w_k1_MAT_s104
                                 + 1.4013 w_k1_MSFT_s104 + 1.22645 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.732938 w_k1_ADM_s104
                                 + 0.556049 w_k1_AXP_s104
                                 + 0.584699 w_k1_DIS_s104
                                 + 0.732102 w_k1_FDX_s104
                                 + 0.649851 w_k1_IBM_s104
                                 + 0.785032 w_k1_INTC_s104
                                 + 0.575595 w_k1_KO_s104
                                 + 0.580762 w_k1_MAT_s104
                                 + 0.761891 w_k1_MSFT_s104
                                 + 0.747731 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.00772 w_k1_ADM_s104
                                 + 1.06048 w_k1_AXP_s104
                                 + 1.14727 w_k1_DIS_s104
                                 + 1.00515 w_k1_FDX_s104
                                 + 1.20871 w_k1_IBM_s104
                                 + 1.10868 w_k1_INTC_s104
                                 + 1.07862 w_k1_KO_s104 + 1.0054 w_k1_MAT_s104
                                 + 1.0236 w_k1_MSFT_s104
                                 + 0.992243 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.689769 w_k1_ADM_s104
                                 + 0.746513 w_k1_AXP_s104
                                 + 0.815633 w_k1_DIS_s104
                                 + 0.76584 w_k1_FDX_s104
                                 + 0.916751 w_k1_IBM_s104
                                 + 0.963388 w_k1_INTC_s104
                                 + 0.70817 w_k1_KO_s104
                                 + 0.750729 w_k1_MAT_s104
                                 + 0.803998 w_k1_MSFT_s104
                                 + 0.777922 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.752646 w_k1_ADM_s104
                                 + 0.534139 w_k1_AXP_s104
                                 + 0.809449 w_k1_DIS_s104
                                 + 0.826072 w_k1_FDX_s104
                                 + 0.676378 w_k1_IBM_s104
                                 + 1.21265 w_k1_INTC_s104
                                 + 0.85613 w_k1_KO_s104
                                 + 0.397821 w_k1_MAT_s104
                                 + 0.939529 w_k1_MSFT_s104
                                 + 0.811472 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.04698 w_k1_ADM_s105
                                 + 1.07975 w_k1_AXP_s105
                                 + 1.33039 w_k1_DIS_s105
                                 + 1.25878 w_k1_FDX_s105
                                 + 1.04973 w_k1_IBM_s105
                                 + 1.20189 w_k1_INTC_s105 + 1.2701 w_k1_KO_s105
                                 + 1.09174 w_k1_MAT_s105
                                 + 1.29982 w_k1_MSFT_s105
                                 + 1.19468 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.27913 w_k1_ADM_s105
                                 + 0.856923 w_k1_AXP_s105
                                 + 0.625618 w_k1_DIS_s105
                                 + 0.952106 w_k1_FDX_s105
                                 + 0.542246 w_k1_IBM_s105
                                 + 1.12913 w_k1_INTC_s105
                                 + 0.61901 w_k1_KO_s105
                                 + 0.508002 w_k1_MAT_s105
                                 + 0.793056 w_k1_MSFT_s105
                                 + 0.929098 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.07669 w_k1_ADM_s105
                                 + 1.30253 w_k1_AXP_s105
                                 + 1.11772 w_k1_DIS_s105
                                 + 1.35603 w_k1_FDX_s105
                                 + 0.759107 w_k1_IBM_s105
                                 + 1.278 w_k1_INTC_s105 + 0.978566 w_k1_KO_s105
                                 + 1.28283 w_k1_MAT_s105
                                 + 1.17439 w_k1_MSFT_s105
                                 + 1.13997 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.892559 w_k1_ADM_s105
                                 + 0.954421 w_k1_AXP_s105
                                 + 0.906304 w_k1_DIS_s105
                                 + 0.982526 w_k1_FDX_s105
                                 + 1.50486 w_k1_IBM_s105
                                 + 1.15371 w_k1_INTC_s105
                                 + 1.26936 w_k1_KO_s105 + 1.82487 w_k1_MAT_s105
                                 + 1.01419 w_k1_MSFT_s105
                                 + 1.13583 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.757177 w_k1_ADM_s105
                                 + 0.589876 w_k1_AXP_s105
                                 + 0.574716 w_k1_DIS_s105
                                 + 0.721289 w_k1_FDX_s105
                                 + 0.668807 w_k1_IBM_s105
                                 + 0.876869 w_k1_INTC_s105
                                 + 0.694952 w_k1_KO_s105
                                 + 0.466929 w_k1_MAT_s105
                                 + 0.713919 w_k1_MSFT_s105
                                 + 0.82441 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.937649 w_k1_ADM_s105
                                 + 0.820802 w_k1_AXP_s105
                                 + 0.966016 w_k1_DIS_s105
                                 + 0.985531 w_k1_FDX_s105
                                 + 1.02699 w_k1_IBM_s105
                                 + 1.16599 w_k1_INTC_s105
                                 + 1.00435 w_k1_KO_s105 + 1.00027 w_k1_MAT_s105
                                 + 1.00886 w_k1_MSFT_s105
                                 + 0.972703 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.977087 w_k1_ADM_s105
                                 + 0.866221 w_k1_AXP_s105
                                 + 0.777206 w_k1_DIS_s105
                                 + 0.879828 w_k1_FDX_s105
                                 + 0.765552 w_k1_IBM_s105
                                 + 1.15993 w_k1_INTC_s105
                                 + 0.837856 w_k1_KO_s105
                                 + 0.548477 w_k1_MAT_s105
                                 + 0.902311 w_k1_MSFT_s105
                                 + 0.999637 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.70439 w_k1_ADM_s105
                                 + 1.29199 w_k1_AXP_s105
                                 + 0.962523 w_k1_DIS_s105
                                 + 1.10728 w_k1_FDX_s105
                                 + 1.17531 w_k1_IBM_s105
                                 + 1.2743 w_k1_INTC_s105 + 1.05604 w_k1_KO_s105
                                 + 1.72 w_k1_MAT_s105 + 1.03309 w_k1_MSFT_s105
                                 + 1.0975 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.791657 w_k1_ADM_s106
                                 + 0.75249 w_k1_AXP_s106
                                 + 0.617905 w_k1_DIS_s106
                                 + 0.782706 w_k1_FDX_s106
                                 + 0.845202 w_k1_IBM_s106
                                 + 0.823609 w_k1_INTC_s106
                                 + 0.71064 w_k1_KO_s106
                                 + 0.713983 w_k1_MAT_s106
                                 + 0.794462 w_k1_MSFT_s106
                                 + 0.711845 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.803319 w_k1_ADM_s106
                                 + 1.23999 w_k1_AXP_s106
                                 + 1.68534 w_k1_DIS_s106
                                 + 1.34824 w_k1_FDX_s106
                                 + 1.49832 w_k1_IBM_s106
                                 + 1.79925 w_k1_INTC_s106
                                 + 1.36798 w_k1_KO_s106 + 1.09941 w_k1_MAT_s106
                                 + 1.51367 w_k1_MSFT_s106
                                 + 1.17502 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.04913 w_k1_ADM_s106
                                 + 1.38215 w_k1_AXP_s106
                                 + 1.29979 w_k1_DIS_s106
                                 + 1.23727 w_k1_FDX_s106
                                 + 1.05971 w_k1_IBM_s106
                                 + 1.2149 w_k1_INTC_s106 + 1.11698 w_k1_KO_s106
                                 + 1.19662 w_k1_MAT_s106
                                 + 1.15364 w_k1_MSFT_s106
                                 + 1.02452 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.901761 w_k1_ADM_s106
                                 + 1.10983 w_k1_AXP_s106 + 1.0463 w_k1_DIS_s106
                                 + 1.04209 w_k1_FDX_s106
                                 + 1.03329 w_k1_IBM_s106
                                 + 1.11633 w_k1_INTC_s106
                                 + 0.964041 w_k1_KO_s106
                                 + 1.04321 w_k1_MAT_s106
                                 + 1.06255 w_k1_MSFT_s106
                                 + 0.998107 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.808669 w_k1_ADM_s106
                                 + 0.819083 w_k1_AXP_s106
                                 + 1.07999 w_k1_DIS_s106
                                 + 0.923337 w_k1_FDX_s106
                                 + 1.14168 w_k1_IBM_s106
                                 + 0.965718 w_k1_INTC_s106
                                 + 0.966038 w_k1_KO_s106
                                 + 0.874638 w_k1_MAT_s106
                                 + 0.996594 w_k1_MSFT_s106
                                 + 0.820585 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.0279 w_k1_ADM_s106
                                 + 1.31556 w_k1_AXP_s106
                                 + 1.47699 w_k1_DIS_s106
                                 + 1.21229 w_k1_FDX_s106
                                 + 1.31008 w_k1_IBM_s106
                                 + 1.24042 w_k1_INTC_s106
                                 + 1.21283 w_k1_KO_s106
                                 + 0.960991 w_k1_MAT_s106
                                 + 1.28781 w_k1_MSFT_s106
                                 + 0.978808 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.755532 w_k1_ADM_s106
                                 + 1.01121 w_k1_AXP_s106
                                 + 0.836691 w_k1_DIS_s106
                                 + 1.03755 w_k1_FDX_s106
                                 + 0.726429 w_k1_IBM_s106
                                 + 0.895007 w_k1_INTC_s106
                                 + 0.764715 w_k1_KO_s106
                                 + 0.796879 w_k1_MAT_s106
                                 + 1.07538 w_k1_MSFT_s106
                                 + 0.815112 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.903364 w_k1_ADM_s106
                                 + 0.949433 w_k1_AXP_s106
                                 + 1.0376 w_k1_DIS_s106 + 1.01783 w_k1_FDX_s106
                                 + 1.16415 w_k1_IBM_s106
                                 + 1.00041 w_k1_INTC_s106
                                 + 1.10949 w_k1_KO_s106
                                 + 0.977311 w_k1_MAT_s106
                                 + 1.002 w_k1_MSFT_s106 + 0.995689 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.709568 w_k1_ADM_s107
                                 + 0.957145 w_k1_AXP_s107
                                 + 0.761096 w_k1_DIS_s107
                                 + 0.903057 w_k1_FDX_s107
                                 + 0.938939 w_k1_IBM_s107
                                 + 0.928802 w_k1_INTC_s107
                                 + 0.821862 w_k1_KO_s107
                                 + 1.14099 w_k1_MAT_s107
                                 + 0.923147 w_k1_MSFT_s107
                                 + 0.875417 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.874328 w_k1_ADM_s107
                                 + 0.750445 w_k1_AXP_s107
                                 + 1.22115 w_k1_DIS_s107
                                 + 0.875743 w_k1_FDX_s107
                                 + 1.07098 w_k1_IBM_s107
                                 + 0.997466 w_k1_INTC_s107
                                 + 1.11277 w_k1_KO_s107 + 0.72136 w_k1_MAT_s107
                                 + 1.00409 w_k1_MSFT_s107
                                 + 1.04736 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.838504 w_k1_ADM_s107
                                 + 0.702684 w_k1_AXP_s107
                                 + 0.807163 w_k1_DIS_s107
                                 + 0.908941 w_k1_FDX_s107
                                 + 0.599803 w_k1_IBM_s107
                                 + 0.916254 w_k1_INTC_s107
                                 + 0.823864 w_k1_KO_s107
                                 + 0.745558 w_k1_MAT_s107
                                 + 1.0571 w_k1_MSFT_s107
                                 + 0.978636 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.12647 w_k1_ADM_s107
                                 + 1.39325 w_k1_AXP_s107 + 1.462 w_k1_DIS_s107
                                 + 1.3802 w_k1_FDX_s107 + 1.17976 w_k1_IBM_s107
                                 + 1.16948 w_k1_INTC_s107 + 1.0783 w_k1_KO_s107
                                 + 1.19927 w_k1_MAT_s107
                                 + 1.22757 w_k1_MSFT_s107 + 1.1618 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.03058 w_k1_ADM_s107
                                 + 1.13883 w_k1_AXP_s107
                                 + 1.65031 w_k1_DIS_s107
                                 + 1.25519 w_k1_FDX_s107
                                 + 0.960534 w_k1_IBM_s107
                                 + 1.23738 w_k1_INTC_s107
                                 + 1.30835 w_k1_KO_s107 + 1.21412 w_k1_MAT_s107
                                 + 1.66981 w_k1_MSFT_s107
                                 + 1.24583 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.862186 w_k1_ADM_s107
                                 + 1.04325 w_k1_AXP_s107
                                 + 0.937564 w_k1_DIS_s107
                                 + 1.13599 w_k1_FDX_s107
                                 + 0.832444 w_k1_IBM_s107
                                 + 0.907989 w_k1_INTC_s107
                                 + 0.81222 w_k1_KO_s107 + 1.13247 w_k1_MAT_s107
                                 + 1.04706 w_k1_MSFT_s107
                                 + 1.02904 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.699321 w_k1_ADM_s107
                                 + 0.508148 w_k1_AXP_s107
                                 + 0.711099 w_k1_DIS_s107
                                 + 0.744063 w_k1_FDX_s107
                                 + 0.618041 w_k1_IBM_s107
                                 + 0.71036 w_k1_INTC_s107
                                 + 0.669762 w_k1_KO_s107
                                 + 0.764506 w_k1_MAT_s107
                                 + 0.847506 w_k1_MSFT_s107
                                 + 0.770184 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.960751 w_k1_ADM_s107
                                 + 0.916561 w_k1_AXP_s107
                                 + 0.941609 w_k1_DIS_s107
                                 + 0.985763 w_k1_FDX_s107
                                 + 1.15021 w_k1_IBM_s107
                                 + 0.877278 w_k1_INTC_s107
                                 + 1.1317 w_k1_KO_s107 + 1.78649 w_k1_MAT_s107
                                 + 1.14531 w_k1_MSFT_s107
                                 + 1.06566 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.1318 w_k1_ADM_s110
                                 + 0.946733 w_k1_AXP_s110
                                 + 1.09481 w_k1_DIS_s110
                                 + 0.961678 w_k1_FDX_s110
                                 + 1.27694 w_k1_IBM_s110
                                 + 1.04575 w_k1_INTC_s110
                                 + 1.02806 w_k1_KO_s110 + 1.04349 w_k1_MAT_s110
                                 + 0.937572 w_k1_MSFT_s110
                                 + 0.938905 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.840938 w_k1_ADM_s110
                                 + 0.677545 w_k1_AXP_s110
                                 + 0.741306 w_k1_DIS_s110
                                 + 0.796325 w_k1_FDX_s110
                                 + 1.09938 w_k1_IBM_s110
                                 + 0.869846 w_k1_INTC_s110
                                 + 0.989053 w_k1_KO_s110
                                 + 1.25777 w_k1_MAT_s110
                                 + 0.911078 w_k1_MSFT_s110
                                 + 0.902596 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.18641 w_k1_ADM_s110
                                 + 0.905427 w_k1_AXP_s110
                                 + 1.03682 w_k1_DIS_s110
                                 + 0.937087 w_k1_FDX_s110
                                 + 0.935088 w_k1_IBM_s110
                                 + 1.10492 w_k1_INTC_s110
                                 + 0.897679 w_k1_KO_s110
                                 + 1.10601 w_k1_MAT_s110
                                 + 1.01249 w_k1_MSFT_s110
                                 + 0.896402 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.960686 w_k1_ADM_s110
                                 + 0.499823 w_k1_AXP_s110
                                 + 0.698061 w_k1_DIS_s110
                                 + 0.803463 w_k1_FDX_s110
                                 + 0.657073 w_k1_IBM_s110
                                 + 0.764192 w_k1_INTC_s110
                                 + 0.801423 w_k1_KO_s110
                                 + 0.568806 w_k1_MAT_s110
                                 + 0.803933 w_k1_MSFT_s110
                                 + 0.758221 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.27526 w_k1_ADM_s110
                                 + 1.11625 w_k1_AXP_s110 + 1.4944 w_k1_DIS_s110
                                 + 1.18248 w_k1_FDX_s110
                                 + 1.49357 w_k1_IBM_s110
                                 + 1.35491 w_k1_INTC_s110
                                 + 1.21512 w_k1_KO_s110 + 1.39898 w_k1_MAT_s110
                                 + 1.31635 w_k1_MSFT_s110
                                 + 1.09876 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.36412 w_k1_ADM_s110
                                 + 1.12855 w_k1_AXP_s110
                                 + 0.771129 w_k1_DIS_s110
                                 + 1.05563 w_k1_FDX_s110 + 0.7508 w_k1_IBM_s110
                                 + 1.02507 w_k1_INTC_s110
                                 + 0.830127 w_k1_KO_s110 + 1.1334 w_k1_MAT_s110
                                 + 0.902023 w_k1_MSFT_s110
                                 + 0.938152 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.729644 w_k1_ADM_s110
                                 + 0.557203 w_k1_AXP_s110
                                 + 0.600092 w_k1_DIS_s110
                                 + 0.609775 w_k1_FDX_s110
                                 + 0.954251 w_k1_IBM_s110
                                 + 0.803844 w_k1_INTC_s110
                                 + 0.759105 w_k1_KO_s110
                                 + 0.782487 w_k1_MAT_s110
                                 + 0.672334 w_k1_MSFT_s110
                                 + 0.731711 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.52364 w_k1_ADM_s110
                                 + 1.1799 w_k1_AXP_s110 + 1.35491 w_k1_DIS_s110
                                 + 1.19804 w_k1_FDX_s110
                                 + 0.863167 w_k1_IBM_s110
                                 + 1.34978 w_k1_INTC_s110 + 1.1283 w_k1_KO_s110
                                 + 1.13915 w_k1_MAT_s110
                                 + 1.18196 w_k1_MSFT_s110
                                 + 1.04798 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.797398 w_k1_ADM_s111
                                 + 0.797351 w_k1_AXP_s111
                                 + 0.684791 w_k1_DIS_s111
                                 + 0.842087 w_k1_FDX_s111
                                 + 0.840567 w_k1_IBM_s111
                                 + 0.974217 w_k1_INTC_s111
                                 + 0.806658 w_k1_KO_s111
                                 + 0.994501 w_k1_MAT_s111
                                 + 0.888067 w_k1_MSFT_s111
                                 + 0.8373 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.39534 w_k1_ADM_s111
                                 + 1.13733 w_k1_AXP_s111
                                 + 1.54822 w_k1_DIS_s111
                                 + 1.22715 w_k1_FDX_s111
                                 + 0.98247 w_k1_IBM_s111
                                 + 1.28812 w_k1_INTC_s111
                                 + 1.16255 w_k1_KO_s111
                                 + 0.912062 w_k1_MAT_s111
                                 + 1.64616 w_k1_MSFT_s111
                                 + 1.09861 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.933717 w_k1_ADM_s111
                                 + 0.978792 w_k1_AXP_s111
                                 + 0.774027 w_k1_DIS_s111
                                 + 0.864068 w_k1_FDX_s111
                                 + 1.15338 w_k1_IBM_s111
                                 + 0.888709 w_k1_INTC_s111
                                 + 0.984932 w_k1_KO_s111
                                 + 1.28112 w_k1_MAT_s111
                                 + 0.966181 w_k1_MSFT_s111
                                 + 0.916858 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.806905 w_k1_ADM_s111
                                 + 0.77071 w_k1_AXP_s111
                                 + 0.711573 w_k1_DIS_s111
                                 + 0.893407 w_k1_FDX_s111
                                 + 0.629084 w_k1_IBM_s111
                                 + 0.94764 w_k1_INTC_s111
                                 + 0.745694 w_k1_KO_s111
                                 + 0.825565 w_k1_MAT_s111
                                 + 1.00626 w_k1_MSFT_s111
                                 + 0.914337 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.933401 w_k1_ADM_s111
                                 + 0.949875 w_k1_AXP_s111
                                 + 1.20048 w_k1_DIS_s111
                                 + 1.07037 w_k1_FDX_s111
                                 + 0.768862 w_k1_IBM_s111
                                 + 0.897062 w_k1_INTC_s111
                                 + 1.08237 w_k1_KO_s111
                                 + 0.851521 w_k1_MAT_s111
                                 + 1.17255 w_k1_MSFT_s111
                                 + 1.03536 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.951012 w_k1_ADM_s111
                                 + 1.17245 w_k1_AXP_s111
                                 + 0.99309 w_k1_DIS_s111
                                 + 1.07823 w_k1_FDX_s111
                                 + 0.894208 w_k1_IBM_s111
                                 + 1.04677 w_k1_INTC_s111
                                 + 0.955515 w_k1_KO_s111
                                 + 1.03743 w_k1_MAT_s111
                                 + 1.07953 w_k1_MSFT_s111
                                 + 0.895194 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04776 w_k1_ADM_s111
                                 + 1.29089 w_k1_AXP_s111
                                 + 1.26741 w_k1_DIS_s111
                                 + 1.08144 w_k1_FDX_s111
                                 + 1.63782 w_k1_IBM_s111
                                 + 1.00047 w_k1_INTC_s111
                                 + 1.29574 w_k1_KO_s111 + 1.91082 w_k1_MAT_s111
                                 + 1.09899 w_k1_MSFT_s111
                                 + 0.956374 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.37403 w_k1_ADM_s111
                                 + 1.26625 w_k1_AXP_s111
                                 + 1.28311 w_k1_DIS_s111
                                 + 1.22236 w_k1_FDX_s111
                                 + 1.16461 w_k1_IBM_s111
                                 + 1.16921 w_k1_INTC_s111
                                 + 1.17489 w_k1_KO_s111 + 1.49645 w_k1_MAT_s111
                                 + 1.45743 w_k1_MSFT_s111 + 1.012 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.18851 w_k1_ADM_s112
                                 + 1.19485 w_k1_AXP_s112
                                 + 1.21661 w_k1_DIS_s112
                                 + 1.17395 w_k1_FDX_s112
                                 + 0.850583 w_k1_IBM_s112
                                 + 0.950404 w_k1_INTC_s112
                                 + 0.976134 w_k1_KO_s112
                                 + 0.761859 w_k1_MAT_s112
                                 + 1.14226 w_k1_MSFT_s112
                                 + 1.13197 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.89893 w_k1_ADM_s112
                                 + 1.21751 w_k1_AXP_s112
                                 + 0.887184 w_k1_DIS_s112
                                 + 0.916371 w_k1_FDX_s112
                                 + 1.09442 w_k1_IBM_s112
                                 + 0.704946 w_k1_INTC_s112
                                 + 0.946248 w_k1_KO_s112
                                 + 1.30964 w_k1_MAT_s112
                                 + 0.777596 w_k1_MSFT_s112
                                 + 0.972601 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.25121 w_k1_ADM_s112
                                 + 1.61372 w_k1_AXP_s112 + 1.8104 w_k1_DIS_s112
                                 + 1.3505 w_k1_FDX_s112 + 1.5685 w_k1_IBM_s112
                                 + 0.855574 w_k1_INTC_s112
                                 + 1.26505 w_k1_KO_s112 + 1.78674 w_k1_MAT_s112
                                 + 1.07438 w_k1_MSFT_s112 + 1.2321 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.892635 w_k1_ADM_s112
                                 + 0.726975 w_k1_AXP_s112
                                 + 0.798757 w_k1_DIS_s112
                                 + 0.840627 w_k1_FDX_s112
                                 + 0.725837 w_k1_IBM_s112
                                 + 0.635774 w_k1_INTC_s112
                                 + 0.827587 w_k1_KO_s112
                                 + 0.683933 w_k1_MAT_s112
                                 + 0.808804 w_k1_MSFT_s112
                                 + 0.975059 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.00786 w_k1_ADM_s112
                                 + 0.870249 w_k1_AXP_s112
                                 + 1.37948 w_k1_DIS_s112
                                 + 1.09519 w_k1_FDX_s112
                                 + 1.25293 w_k1_IBM_s112
                                 + 1.01864 w_k1_INTC_s112
                                 + 1.15237 w_k1_KO_s112 + 1.24657 w_k1_MAT_s112
                                 + 1.0657 w_k1_MSFT_s112 + 1.1044 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.04285 w_k1_ADM_s112
                                 + 1.14249 w_k1_AXP_s112
                                 + 1.26652 w_k1_DIS_s112 + 1.0189 w_k1_FDX_s112
                                 + 1.68743 w_k1_IBM_s112
                                 + 0.890969 w_k1_INTC_s112
                                 + 1.35474 w_k1_KO_s112 + 1.56309 w_k1_MAT_s112
                                 + 0.969725 w_k1_MSFT_s112 + 1.176 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.58169 w_k1_ADM_s112
                                 + 1.39941 w_k1_AXP_s112
                                 + 1.71371 w_k1_DIS_s112
                                 + 1.42865 w_k1_FDX_s112
                                 + 0.954374 w_k1_IBM_s112
                                 + 1.2836 w_k1_INTC_s112 + 1.11607 w_k1_KO_s112
                                 + 0.738647 w_k1_MAT_s112
                                 + 1.21387 w_k1_MSFT_s112
                                 + 1.18017 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.849688 w_k1_ADM_s112
                                 + 0.839363 w_k1_AXP_s112
                                 + 0.872485 w_k1_DIS_s112
                                 + 0.824215 w_k1_FDX_s112
                                 + 1.15439 w_k1_IBM_s112
                                 + 0.698997 w_k1_INTC_s112
                                 + 0.93939 w_k1_KO_s112 + 0.93815 w_k1_MAT_s112
                                 + 0.759018 w_k1_MSFT_s112
                                 + 0.987436 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.756334 w_k1_ADM_s113
                                 + 0.657635 w_k1_AXP_s113
                                 + 0.597182 w_k1_DIS_s113
                                 + 0.61019 w_k1_FDX_s113
                                 + 1.00856 w_k1_IBM_s113
                                 + 0.86706 w_k1_INTC_s113
                                 + 0.848195 w_k1_KO_s113
                                 + 0.841783 w_k1_MAT_s113
                                 + 0.68871 w_k1_MSFT_s113
                                 + 0.737601 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.762847 w_k1_ADM_s113
                                 + 0.704515 w_k1_AXP_s113
                                 + 0.933438 w_k1_DIS_s113
                                 + 0.80428 w_k1_FDX_s113 + 1.2914 w_k1_IBM_s113
                                 + 0.952208 w_k1_INTC_s113
                                 + 1.30127 w_k1_KO_s113 + 1.36889 w_k1_MAT_s113
                                 + 0.896271 w_k1_MSFT_s113
                                 + 0.956913 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.21847 w_k1_ADM_s113
                                 + 1.12459 w_k1_AXP_s113
                                 + 1.41536 w_k1_DIS_s113
                                 + 1.30668 w_k1_FDX_s113
                                 + 0.868305 w_k1_IBM_s113
                                 + 1.34644 w_k1_INTC_s113
                                 + 1.23143 w_k1_KO_s113 + 1.112 w_k1_MAT_s113
                                 + 1.265 w_k1_MSFT_s113 + 0.966885 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.3506 w_k1_ADM_s113
                                 + 1.22803 w_k1_AXP_s113
                                 + 1.73639 w_k1_DIS_s113
                                 + 1.29654 w_k1_FDX_s113
                                 + 1.42611 w_k1_IBM_s113
                                 + 1.32851 w_k1_INTC_s113
                                 + 1.61423 w_k1_KO_s113 + 1.41269 w_k1_MAT_s113
                                 + 1.27996 w_k1_MSFT_s113
                                 + 1.26792 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.17258 w_k1_ADM_s113
                                 + 1.4296 w_k1_AXP_s113 + 1.25356 w_k1_DIS_s113
                                 + 1.21157 w_k1_FDX_s113
                                 + 1.53072 w_k1_IBM_s113
                                 + 1.41461 w_k1_INTC_s113
                                 + 1.52516 w_k1_KO_s113 + 2.27979 w_k1_MAT_s113
                                 + 1.27829 w_k1_MSFT_s113
                                 + 1.17741 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.968158 w_k1_ADM_s113
                                 + 0.871815 w_k1_AXP_s113
                                 + 0.953491 w_k1_DIS_s113
                                 + 0.995109 w_k1_FDX_s113
                                 + 0.78758 w_k1_IBM_s113
                                 + 1.03524 w_k1_INTC_s113
                                 + 0.91949 w_k1_KO_s113
                                 + 0.649997 w_k1_MAT_s113
                                 + 1.05123 w_k1_MSFT_s113
                                 + 0.884906 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.779216 w_k1_ADM_s113
                                 + 0.638304 w_k1_AXP_s113
                                 + 0.631201 w_k1_DIS_s113
                                 + 0.772154 w_k1_FDX_s113
                                 + 0.720833 w_k1_IBM_s113
                                 + 0.894163 w_k1_INTC_s113
                                 + 0.778809 w_k1_KO_s113
                                 + 0.646695 w_k1_MAT_s113
                                 + 0.804567 w_k1_MSFT_s113
                                 + 0.727528 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.952954 w_k1_ADM_s113
                                 + 1.15649 w_k1_AXP_s113
                                 + 1.00421 w_k1_DIS_s113
                                 + 0.983761 w_k1_FDX_s113
                                 + 1.08022 w_k1_IBM_s113
                                 + 0.956093 w_k1_INTC_s113
                                 + 1.11229 w_k1_KO_s113 + 1.66164 w_k1_MAT_s113
                                 + 0.962946 w_k1_MSFT_s113
                                 + 0.938644 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.786626 w_k1_ADM_s114
                                 + 0.635535 w_k1_AXP_s114
                                 + 0.611704 w_k1_DIS_s114
                                 + 0.678357 w_k1_FDX_s114
                                 + 0.959601 w_k1_IBM_s114
                                 + 0.80116 w_k1_INTC_s114
                                 + 0.854533 w_k1_KO_s114
                                 + 0.927512 w_k1_MAT_s114
                                 + 0.819394 w_k1_MSFT_s114
                                 + 0.876465 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.16259 w_k1_ADM_s114
                                 + 1.3101 w_k1_AXP_s114 + 1.30104 w_k1_DIS_s114
                                 + 1.21455 w_k1_FDX_s114
                                 + 1.11668 w_k1_IBM_s114
                                 + 1.22142 w_k1_INTC_s114
                                 + 1.14211 w_k1_KO_s114 + 1.42793 w_k1_MAT_s114
                                 + 1.20476 w_k1_MSFT_s114
                                 + 1.04212 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.36319 w_k1_ADM_s114
                                 + 1.09115 w_k1_AXP_s114 + 1.5755 w_k1_DIS_s114
                                 + 1.20479 w_k1_FDX_s114
                                 + 1.56736 w_k1_IBM_s114
                                 + 1.23989 w_k1_INTC_s114
                                 + 1.70208 w_k1_KO_s114 + 1.58037 w_k1_MAT_s114
                                 + 1.20347 w_k1_MSFT_s114
                                 + 1.15486 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.0562 w_k1_ADM_s114
                                 + 0.932726 w_k1_AXP_s114
                                 + 1.00015 w_k1_DIS_s114
                                 + 0.909749 w_k1_FDX_s114
                                 + 0.906765 w_k1_IBM_s114
                                 + 1.03266 w_k1_INTC_s114
                                 + 0.980582 w_k1_KO_s114
                                 + 0.875979 w_k1_MAT_s114
                                 + 1.01948 w_k1_MSFT_s114
                                 + 0.973537 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.05917 w_k1_ADM_s114
                                 + 1.23811 w_k1_AXP_s114
                                 + 0.752345 w_k1_DIS_s114
                                 + 0.884617 w_k1_FDX_s114
                                 + 1.28055 w_k1_IBM_s114
                                 + 0.934581 w_k1_INTC_s114
                                 + 0.849394 w_k1_KO_s114
                                 + 1.57234 w_k1_MAT_s114
                                 + 0.819747 w_k1_MSFT_s114
                                 + 1.00559 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.23314 w_k1_ADM_s114
                                 + 1.10591 w_k1_AXP_s114
                                 + 1.21225 w_k1_DIS_s114 + 1.1851 w_k1_FDX_s114
                                 + 0.833091 w_k1_IBM_s114
                                 + 1.15084 w_k1_INTC_s114
                                 + 0.980968 w_k1_KO_s114
                                 + 1.21719 w_k1_MAT_s114
                                 + 1.32939 w_k1_MSFT_s114
                                 + 0.987131 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.932679 w_k1_ADM_s114
                                 + 1.00656 w_k1_AXP_s114
                                 + 1.04459 w_k1_DIS_s114
                                 + 0.89379 w_k1_FDX_s114 + 1.2281 w_k1_IBM_s114
                                 + 1.1149 w_k1_INTC_s114 + 1.17143 w_k1_KO_s114
                                 + 1.28905 w_k1_MAT_s114 + 1.029 w_k1_MSFT_s114
                                 + 1.02298 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.978211 w_k1_ADM_s114
                                 + 0.758942 w_k1_AXP_s114
                                 + 0.86085 w_k1_DIS_s114
                                 + 0.877238 w_k1_FDX_s114
                                 + 0.734928 w_k1_IBM_s114
                                 + 0.726685 w_k1_INTC_s114
                                 + 0.897994 w_k1_KO_s114
                                 + 0.639395 w_k1_MAT_s114
                                 + 0.923926 w_k1_MSFT_s114
                                 + 0.882293 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.943162 w_k1_ADM_s115
                                 + 0.738767 w_k1_AXP_s115
                                 + 0.906472 w_k1_DIS_s115
                                 + 0.867349 w_k1_FDX_s115
                                 + 1.08958 w_k1_IBM_s115
                                 + 1.05601 w_k1_INTC_s115
                                 + 1.14152 w_k1_KO_s115
                                 + 0.874036 w_k1_MAT_s115
                                 + 1.07265 w_k1_MSFT_s115
                                 + 1.03916 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.1366 w_k1_ADM_s115
                                 + 1.14922 w_k1_AXP_s115
                                 + 1.38298 w_k1_DIS_s115
                                 + 1.23153 w_k1_FDX_s115
                                 + 1.07179 w_k1_IBM_s115
                                 + 1.11877 w_k1_INTC_s115
                                 + 1.29333 w_k1_KO_s115 + 1.44749 w_k1_MAT_s115
                                 + 1.21997 w_k1_MSFT_s115
                                 + 1.06923 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.24067 w_k1_ADM_s115
                                 + 1.2193 w_k1_AXP_s115 + 1.37975 w_k1_DIS_s115
                                 + 1.22837 w_k1_FDX_s115
                                 + 1.52607 w_k1_IBM_s115
                                 + 0.966237 w_k1_INTC_s115
                                 + 1.38219 w_k1_KO_s115 + 1.89901 w_k1_MAT_s115
                                 + 1.06505 w_k1_MSFT_s115
                                 + 1.21951 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.84156 w_k1_ADM_s115
                                 + 1.02658 w_k1_AXP_s115
                                 + 0.88339 w_k1_DIS_s115
                                 + 0.987284 w_k1_FDX_s115
                                 + 0.906154 w_k1_IBM_s115
                                 + 0.8233 w_k1_INTC_s115 + 1.02144 w_k1_KO_s115
                                 + 1.14796 w_k1_MAT_s115
                                 + 0.970552 w_k1_MSFT_s115
                                 + 0.985305 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.906172 w_k1_ADM_s115
                                 + 0.899675 w_k1_AXP_s115
                                 + 0.890421 w_k1_DIS_s115
                                 + 1.00928 w_k1_FDX_s115
                                 + 0.698117 w_k1_IBM_s115
                                 + 0.783201 w_k1_INTC_s115
                                 + 0.89096 w_k1_KO_s115
                                 + 0.900442 w_k1_MAT_s115
                                 + 0.921129 w_k1_MSFT_s115
                                 + 1.00031 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.889137 w_k1_ADM_s115
                                 + 0.785084 w_k1_AXP_s115
                                 + 1.04786 w_k1_DIS_s115
                                 + 0.930319 w_k1_FDX_s115
                                 + 1.26416 w_k1_IBM_s115
                                 + 0.788795 w_k1_INTC_s115
                                 + 1.29351 w_k1_KO_s115 + 1.41434 w_k1_MAT_s115
                                 + 1.00238 w_k1_MSFT_s115
                                 + 1.04151 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.769759 w_k1_ADM_s115
                                 + 0.65371 w_k1_AXP_s115
                                 + 0.648916 w_k1_DIS_s115
                                 + 0.753378 w_k1_FDX_s115
                                 + 0.675441 w_k1_IBM_s115
                                 + 0.722615 w_k1_INTC_s115
                                 + 0.813381 w_k1_KO_s115
                                 + 0.631395 w_k1_MAT_s115
                                 + 0.785756 w_k1_MSFT_s115
                                 + 0.947284 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.718197 w_k1_ADM_s115
                                 + 0.67295 w_k1_AXP_s115
                                 + 0.706626 w_k1_DIS_s115
                                 + 0.799308 w_k1_FDX_s115
                                 + 0.929985 w_k1_IBM_s115
                                 + 0.690231 w_k1_INTC_s115
                                 + 0.84227 w_k1_KO_s115 + 1.03538 w_k1_MAT_s115
                                 + 0.787464 w_k1_MSFT_s115
                                 + 1.05562 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.865327 w_k1_ADM_s116
                                 + 1.06625 w_k1_AXP_s116
                                 + 1.33061 w_k1_DIS_s116
                                 + 1.01834 w_k1_FDX_s116
                                 + 1.30118 w_k1_IBM_s116
                                 + 0.954305 w_k1_INTC_s116
                                 + 1.20716 w_k1_KO_s116 + 1.02522 w_k1_MAT_s116
                                 + 1.04631 w_k1_MSFT_s116
                                 + 1.04727 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2311 w_k1_ADM_s116
                                 + 1.39586 w_k1_AXP_s116
                                 + 1.72841 w_k1_DIS_s116
                                 + 1.35718 w_k1_FDX_s116
                                 + 1.06055 w_k1_IBM_s116
                                 + 1.31549 w_k1_INTC_s116
                                 + 1.17158 w_k1_KO_s116 + 1.09528 w_k1_MAT_s116
                                 + 1.15781 w_k1_MSFT_s116
                                 + 1.27741 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.805201 w_k1_ADM_s116
                                 + 0.934283 w_k1_AXP_s116
                                 + 1.16698 w_k1_DIS_s116
                                 + 0.995885 w_k1_FDX_s116
                                 + 0.93688 w_k1_IBM_s116
                                 + 0.880616 w_k1_INTC_s116
                                 + 0.994004 w_k1_KO_s116
                                 + 1.00041 w_k1_MAT_s116
                                 + 1.00245 w_k1_MSFT_s116
                                 + 1.06643 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.731755 w_k1_ADM_s116
                                 + 0.732036 w_k1_AXP_s116
                                 + 0.737252 w_k1_DIS_s116
                                 + 0.684103 w_k1_FDX_s116
                                 + 1.16858 w_k1_IBM_s116
                                 + 0.7805 w_k1_INTC_s116 + 1.07141 w_k1_KO_s116
                                 + 0.923196 w_k1_MAT_s116
                                 + 0.84164 w_k1_MSFT_s116
                                 + 0.901659 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.719218 w_k1_ADM_s116
                                 + 0.679006 w_k1_AXP_s116
                                 + 0.724144 w_k1_DIS_s116
                                 + 0.741246 w_k1_FDX_s116
                                 + 0.738975 w_k1_IBM_s116
                                 + 0.785412 w_k1_INTC_s116
                                 + 0.910197 w_k1_KO_s116
                                 + 0.799921 w_k1_MAT_s116
                                 + 0.833119 w_k1_MSFT_s116
                                 + 0.929317 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.877736 w_k1_ADM_s116
                                 + 1.04447 w_k1_AXP_s116
                                 + 0.885492 w_k1_DIS_s116
                                 + 0.985958 w_k1_FDX_s116
                                 + 0.792595 w_k1_IBM_s116
                                 + 1.04578 w_k1_INTC_s116
                                 + 0.842108 w_k1_KO_s116
                                 + 0.825085 w_k1_MAT_s116
                                 + 0.877231 w_k1_MSFT_s116
                                 + 1.02177 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.940334 w_k1_ADM_s116
                                 + 1.10547 w_k1_AXP_s116
                                 + 1.03246 w_k1_DIS_s116
                                 + 0.856775 w_k1_FDX_s116
                                 + 1.55024 w_k1_IBM_s116
                                 + 0.944857 w_k1_INTC_s116
                                 + 1.13423 w_k1_KO_s116 + 1.58793 w_k1_MAT_s116
                                 + 0.917222 w_k1_MSFT_s116
                                 + 1.11413 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.995873 w_k1_ADM_s116
                                 + 1.27774 w_k1_AXP_s116
                                 + 1.05187 w_k1_DIS_s116
                                 + 1.07503 w_k1_FDX_s116
                                 + 1.11851 w_k1_IBM_s116
                                 + 0.997892 w_k1_INTC_s116
                                 + 1.05662 w_k1_KO_s116 + 1.37607 w_k1_MAT_s116
                                 + 0.986423 w_k1_MSFT_s116
                                 + 1.11439 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.634954 w_k1_ADM_s117
                                 + 0.752332 w_k1_AXP_s117
                                 + 0.822646 w_k1_DIS_s117
                                 + 0.756454 w_k1_FDX_s117
                                 + 1.21863 w_k1_IBM_s117
                                 + 0.873025 w_k1_INTC_s117
                                 + 1.02834 w_k1_KO_s117 + 1.03721 w_k1_MAT_s117
                                 + 0.934352 w_k1_MSFT_s117
                                 + 0.890367 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.00711 w_k1_ADM_s117
                                 + 1.17158 w_k1_AXP_s117
                                 + 1.40863 w_k1_DIS_s117
                                 + 1.23807 w_k1_FDX_s117
                                 + 0.827846 w_k1_IBM_s117
                                 + 1.16115 w_k1_INTC_s117
                                 + 0.967622 w_k1_KO_s117
                                 + 1.18846 w_k1_MAT_s117
                                 + 1.3381 w_k1_MSFT_s117 + 1.111 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.750696 w_k1_ADM_s117
                                 + 1.03932 w_k1_AXP_s117
                                 + 1.08572 w_k1_DIS_s117
                                 + 0.954685 w_k1_FDX_s117
                                 + 0.948091 w_k1_IBM_s117
                                 + 0.969258 w_k1_INTC_s117
                                 + 0.973914 w_k1_KO_s117
                                 + 1.13472 w_k1_MAT_s117
                                 + 1.03683 w_k1_MSFT_s117
                                 + 0.994437 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.23132 w_k1_ADM_s117
                                 + 1.42652 w_k1_AXP_s117
                                 + 1.65928 w_k1_DIS_s117
                                 + 1.49593 w_k1_FDX_s117
                                 + 0.812921 w_k1_IBM_s117
                                 + 1.23766 w_k1_INTC_s117
                                 + 1.10127 w_k1_KO_s117
                                 + 0.756038 w_k1_MAT_s117
                                 + 1.23506 w_k1_MSFT_s117
                                 + 1.19085 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.977502 w_k1_ADM_s117
                                 + 1.37643 w_k1_AXP_s117
                                 + 1.37247 w_k1_DIS_s117
                                 + 1.05523 w_k1_FDX_s117
                                 + 1.47449 w_k1_IBM_s117
                                 + 1.03988 w_k1_INTC_s117
                                 + 1.25512 w_k1_KO_s117 + 1.84278 w_k1_MAT_s117
                                 + 1.17237 w_k1_MSFT_s117
                                 + 1.19375 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.778611 w_k1_ADM_s117
                                 + 1.02933 w_k1_AXP_s117
                                 + 0.677225 w_k1_DIS_s117
                                 + 0.793683 w_k1_FDX_s117
                                 + 0.862199 w_k1_IBM_s117
                                 + 0.792818 w_k1_INTC_s117
                                 + 0.774645 w_k1_KO_s117
                                 + 1.35091 w_k1_MAT_s117
                                 + 0.845192 w_k1_MSFT_s117
                                 + 0.890233 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.79744 w_k1_ADM_s117
                                 + 0.744778 w_k1_AXP_s117
                                 + 0.825189 w_k1_DIS_s117
                                 + 0.924242 w_k1_FDX_s117
                                 + 0.637258 w_k1_IBM_s117
                                 + 0.940287 w_k1_INTC_s117
                                 + 0.751533 w_k1_KO_s117
                                 + 0.636038 w_k1_MAT_s117
                                 + 0.946237 w_k1_MSFT_s117
                                 + 0.910928 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.20351 w_k1_ADM_s117
                                 + 1.28302 w_k1_AXP_s117
                                 + 1.42839 w_k1_DIS_s117
                                 + 1.16452 w_k1_FDX_s117
                                 + 1.14378 w_k1_IBM_s117
                                 + 1.20344 w_k1_INTC_s117
                                 + 1.27417 w_k1_KO_s117 + 1.48429 w_k1_MAT_s117
                                 + 1.20731 w_k1_MSFT_s117
                                 + 1.11166 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0171030018335481 w_k1_ADM_s000
                                 + 0.0299146011746186 w_k1_ADM_s001
                                 + 0.0219949388843414 w_k1_ADM_s002
                                 + 0.0530910551965709 w_k1_ADM_s003
                                 + 0.0250298643791711 w_k1_ADM_s004
                                 + 0.0180546145359237 w_k1_ADM_s005
                                 + 0.0356316687796593 w_k1_ADM_s006
                                 + 0.0188207964581676 w_k1_ADM_s007
                                 + 0.0458016367426531 w_k1_ADM_s010
                                 + 0.0349023950882221 w_k1_ADM_s011
                                 + 0.0356251340734042 w_k1_ADM_s012
                                 + 0.0296437185713807 w_k1_ADM_s013
                                 + 0.0348574236591691 w_k1_ADM_s014
                                 + 0.032559151437917 w_k1_ADM_s015
                                 + 0.0354731764492178 w_k1_ADM_s016
                                 + 0.03140044954827 w_k1_ADM_s017
                                 + 0.00976668808571313 w_k1_ADM_s100
                                 + 0.0278692989507324 w_k1_ADM_s101
                                 + 0.0234697060260038 w_k1_ADM_s102
                                 + 0.023364970897081 w_k1_ADM_s103
                                 + 0.0233247058939834 w_k1_ADM_s104
                                 + 0.017676661101905 w_k1_ADM_s105
                                 + 0.00625809572529363 w_k1_ADM_s106
                                 + 0.00555877352745602 w_k1_ADM_s107
                                 + 0.0337118245416638 w_k1_ADM_s110
                                 + 0.0362490626719758 w_k1_ADM_s111
                                 + 0.0450162268558631 w_k1_ADM_s112
                                 + 0.0476678450531266 w_k1_ADM_s113
                                 + 0.0372001455663721 w_k1_ADM_s114
                                 + 0.0281545084235474 w_k1_ADM_s115
                                 + 0.0451240736262973 w_k1_ADM_s116
                                 + 0.0882890998487837 w_k1_ADM_s117
                                 + 0.0173432085731259 w_k1_AXP_s000
                                 + 0.0269356496584512 w_k1_AXP_s001
                                 + 0.0206060169777082 w_k1_AXP_s002
                                 + 0.0513084604131766 w_k1_AXP_s003
                                 + 0.0208950171866609 w_k1_AXP_s004
                                 + 0.0201433461503647 w_k1_AXP_s005
                                 + 0.0419060162942383 w_k1_AXP_s006
                                 + 0.0193907904639243 w_k1_AXP_s007
                                 + 0.0492385095071664 w_k1_AXP_s010
                                 + 0.035032441826402 w_k1_AXP_s011
                                 + 0.0382460012294919 w_k1_AXP_s012
                                 + 0.0306529001935694 w_k1_AXP_s013
                                 + 0.0348645206693232 w_k1_AXP_s014
                                 + 0.0298715700284888 w_k1_AXP_s015
                                 + 0.0414888850023039 w_k1_AXP_s016
                                 + 0.0260500104672826 w_k1_AXP_s017
                                 + 0.00848455013931272 w_k1_AXP_s100
                                 + 0.0281599727612911 w_k1_AXP_s101
                                 + 0.0265189814723889 w_k1_AXP_s102
                                 + 0.0291916278322263 w_k1_AXP_s103
                                 + 0.0247345839382635 w_k1_AXP_s104
                                 + 0.018149118384201 w_k1_AXP_s105
                                 + 0.00745066424460659 w_k1_AXP_s106
                                 + 0.00558671688645161 w_k1_AXP_s107
                                 + 0.0258205466086687 w_k1_AXP_s110
                                 + 0.0375471559354143 w_k1_AXP_s111
                                 + 0.0448966534174479 w_k1_AXP_s112
                                 + 0.0466637825967957 w_k1_AXP_s113
                                 + 0.0346144605807875 w_k1_AXP_s114
                                 + 0.0279108841481116 w_k1_AXP_s115
                                 + 0.0517889665650941 w_k1_AXP_s116
                                 + 0.108173153752121 w_k1_AXP_s117
                                 + 0.0161793696262599 w_k1_DIS_s000
                                 + 0.0296917875239294 w_k1_DIS_s001
                                 + 0.0191677783932294 w_k1_DIS_s002
                                 + 0.0538297910309952 w_k1_DIS_s003
                                 + 0.0218722686731272 w_k1_DIS_s004
                                 + 0.0188005069444327 w_k1_DIS_s005
                                 + 0.0390761054141464 w_k1_DIS_s006
                                 + 0.019352884082351 w_k1_DIS_s007
                                 + 0.0517635925908385 w_k1_DIS_s010
                                 + 0.037410403835151 w_k1_DIS_s011
                                 + 0.0359157227595325 w_k1_DIS_s012
                                 + 0.0290254479638683 w_k1_DIS_s013
                                 + 0.0345435047307787 w_k1_DIS_s014
                                 + 0.0329030337390137 w_k1_DIS_s015
                                 + 0.0417917699947564 w_k1_DIS_s016
                                 + 0.0274118386958326 w_k1_DIS_s017
                                 + 0.00912607967112126 w_k1_DIS_s100
                                 + 0.0272310196173924 w_k1_DIS_s101
                                 + 0.0274828370941407 w_k1_DIS_s102
                                 + 0.0276183446432649 w_k1_DIS_s103
                                 + 0.0264821219231538 w_k1_DIS_s104
                                 + 0.0180525822574131 w_k1_DIS_s105
                                 + 0.00760252405254855 w_k1_DIS_s106
                                 + 0.0064137040420543 w_k1_DIS_s107
                                 + 0.0303099007843555 w_k1_DIS_s110
                                 + 0.036533334961658 w_k1_DIS_s111
                                 + 0.0483863338027546 w_k1_DIS_s112
                                 + 0.0518520081025156 w_k1_DIS_s113
                                 + 0.0374636587167643 w_k1_DIS_s114
                                 + 0.0303180810218911 w_k1_DIS_s115
                                 + 0.0556437196742357 w_k1_DIS_s116
                                 + 0.109755472825247 w_k1_DIS_s117
                                 + 0.0160063832715899 w_k1_FDX_s000
                                 + 0.028507528902264 w_k1_FDX_s001
                                 + 0.0196676017619185 w_k1_FDX_s002
                                 + 0.0515785553923905 w_k1_FDX_s003
                                 + 0.0212793231676831 w_k1_FDX_s004
                                 + 0.0183480258709673 w_k1_FDX_s005
                                 + 0.0375325384035256 w_k1_FDX_s006
                                 + 0.0193803289444611 w_k1_FDX_s007
                                 + 0.0484393074411638 w_k1_FDX_s010
                                 + 0.0351058376994997 w_k1_FDX_s011
                                 + 0.0345745727600658 w_k1_FDX_s012
                                 + 0.0275286661051227 w_k1_FDX_s013
                                 + 0.0322308702862456 w_k1_FDX_s014
                                 + 0.0319750408023037 w_k1_FDX_s015
                                 + 0.0395693281013652 w_k1_FDX_s016
                                 + 0.0289152725169523 w_k1_FDX_s017
                                 + 0.00878071431886288 w_k1_FDX_s100
                                 + 0.0270569161512893 w_k1_FDX_s101
                                 + 0.0274026444348964 w_k1_FDX_s102
                                 + 0.0263274435107806 w_k1_FDX_s103
                                 + 0.0250057751692235 w_k1_FDX_s104
                                 + 0.0195248978769945 w_k1_FDX_s105
                                 + 0.0074863935252681 w_k1_FDX_s106
                                 + 0.00628902412801655 w_k1_FDX_s107
                                 + 0.0282259887234782 w_k1_FDX_s110
                                 + 0.0364773449658565 w_k1_FDX_s111
                                 + 0.044314947032683 w_k1_FDX_s112
                                 + 0.0486555160351125 w_k1_FDX_s113
                                 + 0.0345318619959011 w_k1_FDX_s114
                                 + 0.0298730128965029 w_k1_FDX_s115
                                 + 0.0492590005840009 w_k1_FDX_s116
                                 + 0.0996402993249844 w_k1_FDX_s117
                                 + 0.0167147060210683 w_k1_IBM_s000
                                 + 0.0301892331435867 w_k1_IBM_s001
                                 + 0.0196605445249005 w_k1_IBM_s002
                                 + 0.0513616047486265 w_k1_IBM_s003
                                 + 0.0227168237681731 w_k1_IBM_s004
                                 + 0.0201352513596141 w_k1_IBM_s005
                                 + 0.0398488981994076 w_k1_IBM_s006
                                 + 0.0190079270181359 w_k1_IBM_s007
                                 + 0.0483424076997092 w_k1_IBM_s010
                                 + 0.0362584567836074 w_k1_IBM_s011
                                 + 0.0351790734329185 w_k1_IBM_s012
                                 + 0.0289110050866808 w_k1_IBM_s013
                                 + 0.0313386000449536 w_k1_IBM_s014
                                 + 0.0325521333984942 w_k1_IBM_s015
                                 + 0.0408446001892877 w_k1_IBM_s016
                                 + 0.0297264147906072 w_k1_IBM_s017
                                 + 0.00921751267036176 w_k1_IBM_s100
                                 + 0.0271052647289126 w_k1_IBM_s101
                                 + 0.0278179024215902 w_k1_IBM_s102
                                 + 0.026501498646425 w_k1_IBM_s103
                                 + 0.0256673616631709 w_k1_IBM_s104
                                 + 0.0194848994768049 w_k1_IBM_s105
                                 + 0.00746819224631804 w_k1_IBM_s106
                                 + 0.00569090444174307 w_k1_IBM_s107
                                 + 0.0299717408293347 w_k1_IBM_s110
                                 + 0.035850524335404 w_k1_IBM_s111
                                 + 0.0463032577605871 w_k1_IBM_s112
                                 + 0.051881495028382 w_k1_IBM_s113
                                 + 0.0363815667724174 w_k1_IBM_s114
                                 + 0.030672397943974 w_k1_IBM_s115
                                 + 0.053097901102011 w_k1_IBM_s116
                                 + 0.103144146923799 w_k1_IBM_s117
                                 + 0.0163330190744472 w_k1_INTC_s000
                                 + 0.0284308167989282 w_k1_INTC_s001
                                 + 0.0185362896260092 w_k1_INTC_s002
                                 + 0.0479704475899732 w_k1_INTC_s003
                                 + 0.022860529221301 w_k1_INTC_s004
                                 + 0.0177520652640634 w_k1_INTC_s005
                                 + 0.0359091781543487 w_k1_INTC_s006
                                 + 0.0172809398850035 w_k1_INTC_s007
                                 + 0.0494845538643239 w_k1_INTC_s010
                                 + 0.0388370517849309 w_k1_INTC_s011
                                 + 0.0376801257413182 w_k1_INTC_s012
                                 + 0.0260606946978311 w_k1_INTC_s013
                                 + 0.0315803068933076 w_k1_INTC_s014
                                 + 0.029130995907024 w_k1_INTC_s015
                                 + 0.0371740910138917 w_k1_INTC_s016
                                 + 0.0266955040837242 w_k1_INTC_s017
                                 + 0.00844280378700881 w_k1_INTC_s100
                                 + 0.0301167437784537 w_k1_INTC_s101
                                 + 0.0249760862876794 w_k1_INTC_s102
                                 + 0.0262590436443244 w_k1_INTC_s103
                                 + 0.0298717983013412 w_k1_INTC_s104
                                 + 0.0219348891392577 w_k1_INTC_s105
                                 + 0.00769683700477226 w_k1_INTC_s106
                                 + 0.00592412586001036 w_k1_INTC_s107
                                 + 0.0312604030558245 w_k1_INTC_s110
                                 + 0.0358166449559775 w_k1_INTC_s111
                                 + 0.0357941380422136 w_k1_INTC_s112
                                 + 0.0528523794432146 w_k1_INTC_s113
                                 + 0.0358099081229436 w_k1_INTC_s114
                                 + 0.0262985941574305 w_k1_INTC_s115
                                 + 0.0489289366458646 w_k1_INTC_s116
                                 + 0.100055525570938 w_k1_INTC_s117
                                 + 0.015620445965071 w_k1_KO_s000
                                 + 0.0263368073746423 w_k1_KO_s001
                                 + 0.019859540300712 w_k1_KO_s002
                                 + 0.0530512852675165 w_k1_KO_s003
                                 + 0.0207187492344865 w_k1_KO_s004
                                 + 0.0174126653340637 w_k1_KO_s005
                                 + 0.0354800007875272 w_k1_KO_s006
                                 + 0.0184013477650549 w_k1_KO_s007
                                 + 0.0479995086954069 w_k1_KO_s010
                                 + 0.0394176504134134 w_k1_KO_s011
                                 + 0.0354198433201559 w_k1_KO_s012
                                 + 0.027761335012788 w_k1_KO_s013
                                 + 0.0289028385008581 w_k1_KO_s014
                                 + 0.0311533346800043 w_k1_KO_s015
                                 + 0.042270997390491 w_k1_KO_s016
                                 + 0.0285640052815479 w_k1_KO_s017
                                 + 0.00908137531955786 w_k1_KO_s100
                                 + 0.0266849877878657 w_k1_KO_s101
                                 + 0.0256364739130082 w_k1_KO_s102
                                 + 0.0273509219493102 w_k1_KO_s103
                                 + 0.0242258133620325 w_k1_KO_s104
                                 + 0.0196629458211619 w_k1_KO_s105
                                 + 0.00697252029567119 w_k1_KO_s106
                                 + 0.00603928221628467 w_k1_KO_s107
                                 + 0.0285838690524862 w_k1_KO_s110
                                 + 0.0361043776670493 w_k1_KO_s111
                                 + 0.0439694191556424 w_k1_KO_s112
                                 + 0.0565590412583497 w_k1_KO_s113
                                 + 0.0379028549598633 w_k1_KO_s114
                                 + 0.0330789338482485 w_k1_KO_s115
                                 + 0.0524756499817678 w_k1_KO_s116
                                 + 0.101249121363692 w_k1_KO_s117
                                 + 0.0139837030908759 w_k1_MAT_s000
                                 + 0.0282255982303062 w_k1_MAT_s001
                                 + 0.0183152252102433 w_k1_MAT_s002
                                 + 0.051558274817068 w_k1_MAT_s003
                                 + 0.0257958426355658 w_k1_MAT_s004
                                 + 0.0236355660962628 w_k1_MAT_s005
                                 + 0.0462047761728146 w_k1_MAT_s006
                                 + 0.01485586903283 w_k1_MAT_s007
                                 + 0.0557000229502132 w_k1_MAT_s010
                                 + 0.032431244741813 w_k1_MAT_s011
                                 + 0.0338396171360386 w_k1_MAT_s012
                                 + 0.0292730616996594 w_k1_MAT_s013
                                 + 0.0249316170118565 w_k1_MAT_s014
                                 + 0.0308157878124875 w_k1_MAT_s015
                                 + 0.0376637740313551 w_k1_MAT_s016
                                 + 0.0281042732316694 w_k1_MAT_s017
                                 + 0.00723078512973001 w_k1_MAT_s100
                                 + 0.0231091086083875 w_k1_MAT_s101
                                 + 0.0328626444976616 w_k1_MAT_s102
                                 + 0.0293185350442507 w_k1_MAT_s103
                                 + 0.0231362041682306 w_k1_MAT_s104
                                 + 0.0214097188798801 w_k1_MAT_s105
                                 + 0.00666707364962511 w_k1_MAT_s106
                                 + 0.00685314275771343 w_k1_MAT_s107
                                 + 0.0305214174526422 w_k1_MAT_s110
                                 + 0.0420362466366154 w_k1_MAT_s111
                                 + 0.04401834336821 w_k1_MAT_s112
                                 + 0.0595449834643776 w_k1_MAT_s113
                                 + 0.0396407951971946 w_k1_MAT_s114
                                 + 0.0361150033902174 w_k1_MAT_s115
                                 + 0.0526506329185784 w_k1_MAT_s116
                                 + 0.124926566240125 w_k1_MAT_s117
                                 + 0.0152569097766379 w_k1_MSFT_s000
                                 + 0.0307935083827818 w_k1_MSFT_s001
                                 + 0.0185276403483486 w_k1_MSFT_s002
                                 + 0.0502842021209214 w_k1_MSFT_s003
                                 + 0.0219990870782894 w_k1_MSFT_s004
                                 + 0.0185669385814385 w_k1_MSFT_s005
                                 + 0.0354137495543379 w_k1_MSFT_s006
                                 + 0.0176878587386049 w_k1_MSFT_s007
                                 + 0.0496260588218585 w_k1_MSFT_s010
                                 + 0.036107727940096 w_k1_MSFT_s011
                                 + 0.0345430874963262 w_k1_MSFT_s012
                                 + 0.0254884509133299 w_k1_MSFT_s013
                                 + 0.0295708499908342 w_k1_MSFT_s014
                                 + 0.0329561287269886 w_k1_MSFT_s015
                                 + 0.0373429701342009 w_k1_MSFT_s016
                                 + 0.0270611741831974 w_k1_MSFT_s017
                                 + 0.00828447968581592 w_k1_MSFT_s100
                                 + 0.0269967869095904 w_k1_MSFT_s101
                                 + 0.0270693761039525 w_k1_MSFT_s102
                                 + 0.0291367028355936 w_k1_MSFT_s103
                                 + 0.0260206794972433 w_k1_MSFT_s104
                                 + 0.0194093752254664 w_k1_MSFT_s105
                                 + 0.00768639086963502 w_k1_MSFT_s106
                                 + 0.00685890622081439 w_k1_MSFT_s107
                                 + 0.0292146633014888 w_k1_MSFT_s110
                                 + 0.0404149466786498 w_k1_MSFT_s111
                                 + 0.0408074619098249 w_k1_MSFT_s112
                                 + 0.0501996242925448 w_k1_MSFT_s113
                                 + 0.0370631709973804 w_k1_MSFT_s114
                                 + 0.0297876163295747 w_k1_MSFT_s115
                                 + 0.0485282888781954 w_k1_MSFT_s116
                                 + 0.106988337621783 w_k1_MSFT_s117
                                 + 0.0167825963285373 w_k1_PG_s000
                                 + 0.0284306928695492 w_k1_PG_s001
                                 + 0.0194341459088225 w_k1_PG_s002
                                 + 0.0559788969608846 w_k1_PG_s003
                                 + 0.0217287106117742 w_k1_PG_s004
                                 + 0.0186925251995291 w_k1_PG_s005
                                 + 0.0367495306999878 w_k1_PG_s006
                                 + 0.0183329989578183 w_k1_PG_s007
                                 + 0.0512664834245731 w_k1_PG_s010
                                 + 0.0382658598720789 w_k1_PG_s011
                                 + 0.0332809400304269 w_k1_PG_s012
                                 + 0.0269960625283979 w_k1_PG_s013
                                 + 0.0323390592166041 w_k1_PG_s014
                                 + 0.030116324342092 w_k1_PG_s015
                                 + 0.0376936267998849 w_k1_PG_s016
                                 + 0.0272598141758469 w_k1_PG_s017
                                 + 0.00918388687146613 w_k1_PG_s100
                                 + 0.024402176325664 w_k1_PG_s101
                                 + 0.0268997925109194 w_k1_PG_s102
                                 + 0.0266208960756353 w_k1_PG_s103
                                 + 0.0246944104680203 w_k1_PG_s104
                                 + 0.020002143619285 w_k1_PG_s105
                                 + 0.0064732771325078 w_k1_PG_s106
                                 + 0.00639204452986386 w_k1_PG_s107
                                 + 0.0270357282805976 w_k1_PG_s110
                                 + 0.0333196723474452 w_k1_PG_s111
                                 + 0.0466730982984603 w_k1_PG_s112
                                 + 0.046279107769465 w_k1_PG_s113
                                 + 0.034414445232005 w_k1_PG_s114
                                 + 0.0307739294787455 w_k1_PG_s115
                                 + 0.053415740277146 w_k1_PG_s116
                                 + 0.102911270671377 w_k1_PG_s117 >= 1.229
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
