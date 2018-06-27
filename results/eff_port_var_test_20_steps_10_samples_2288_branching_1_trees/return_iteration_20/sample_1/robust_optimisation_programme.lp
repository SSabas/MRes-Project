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
 w_k1_ADM_s0_return_constraint:  1.19509 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.972873 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.912644 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.09441 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.13176 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.895371 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.825413 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.962234 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.20281 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.956018 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.19913 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.696836 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.0797 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.957742 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.05182 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.782256 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.20778 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.33158 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.880283 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.06874 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.17108 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.13684 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.947651 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.38736 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02383 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.777887 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.905481 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.13679 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.20445 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.11161 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.833578 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.955392 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.833434 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.23038 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.24583 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06553 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.00528 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.06806 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26345 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.980972 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.83412 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.23102 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.23528 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.841083 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.889688 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.878546 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.22332 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.772811 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.37847 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.641256 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.19187 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.15068 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.987231 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.702053 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.09229 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.47158 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.823406 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.981849 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.996433 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1386 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.08997 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.55605 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.947961 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.796858 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.790931 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.47644 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.35332 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.811493 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.673613 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.845042 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.711457 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.34506 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.44401 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.16726 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.948137 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.2857 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32276 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.949389 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.82178 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.30749 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.39953 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.828545 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.700544 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.679924 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.21244 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.892873 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.37476 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.634772 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.2712 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.945367 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.0488 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.639924 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.43975 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.81775 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.794905 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.15984 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.880522 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.16715 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.917052 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.60056 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.988241 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.525243 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.839048 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.66059 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.23791 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.79547 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.659553 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.06773 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.647239 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.58091 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.61307 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.959921 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.991252 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.914369 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.25646 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.937333 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.887714 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.16013 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2395 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.90742 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.785037 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.933501 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.01023 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.94172 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.38624 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.73888 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.18647 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.04954 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.973662 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.707953 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.34095 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.40623 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.893183 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.10555 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.983575 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.948948 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.830967 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.43147 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.0365 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.706398 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.802039 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.19889 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.1182 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.936216 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.808374 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.910244 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.734016 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.38909 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.25455 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.03002 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.07406 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.06496 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.99673 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.03668 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.817063 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.24288 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.34746 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.844734 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.15188 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.613771 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.65141 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.724012 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.835587 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.703095 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.14823 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.8807 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.27329 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.773513 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.744109 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.36001 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.914428 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.857427 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.649537 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.99938 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.29846 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.867375 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.743719 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.918866 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.875727 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.91331 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.07147 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.582071 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.605109 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.28231 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.897508 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.981117 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.77057 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.23998 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.683444 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.925562 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.24847 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.947115 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.85276 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.0092 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.32825 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.963587 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.796502 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.911392 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.00082 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.0532 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.32666 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.873996 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.11451 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.04778 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.922949 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.692618 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.3581 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.45808 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.901452 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.16828 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.924656 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.871206 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.775639 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01552 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.8092 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.693737 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.871617 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.04606 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.958243 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.861283 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.810885 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.814126 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.795289 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.2615 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.27493 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.94715 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.987812 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.12163 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13215 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.954511 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.923053 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.20956 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.40733 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.942838 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.810193 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.644 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.18556 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.750872 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.02737 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.653239 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.0287 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.841097 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.19634 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.758779 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.00673 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.49491 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.830204 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.0124 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.685064 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.28005 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.990177 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.14255 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.827034 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.749648 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.813791 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.42913 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.15227 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.712121 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.646717 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.17848 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.686344 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.30747 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.38367 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.989949 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.797816 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.797467 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.05498 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.933625 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.709013 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.46085 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.55258 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.876845 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.46176 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.762302 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.72182 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.593358 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.17468 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.642061 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.2293 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.902708 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.28096 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.661074 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.633749 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.44516 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.79731 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.73905 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.45586 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.8743 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.21484 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.07097 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.816572 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.902124 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.725361 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.88336 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.31451 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.557308 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.748307 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.37066 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.856579 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.990305 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.64535 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.3582 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.593017 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.928598 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21781 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.969709 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.847648 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.12251 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.25237 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.908684 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.827355 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.856996 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.1115 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.988168 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.2976 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.833785 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08188 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.956096 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.938227 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.703542 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.23054 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.46667 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.895551 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.10406 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.860052 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.893399 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.815413 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.21047 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.958917 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.701589 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.784553 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.16614 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.98284 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.852684 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.797968 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.945286 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.833002 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.21141 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.18082 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.933808 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.967337 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.931807 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10882 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.964028 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.920393 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12133 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.23088 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.951026 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.817843 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.947071 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.15698 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.871178 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10341 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.75269 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.06403 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.985689 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.12499 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.811284 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.16018 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.29504 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.824216 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.01619 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.905393 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.01934 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.957729 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.14071 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.930601 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.796635 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.829741 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.2602 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.10265 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.870303 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.819421 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.02752 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.8268 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.19505 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.16254 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.00234 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.934923 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.946302 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0829998569348984 z_k1_s0000
                                 - 0.113999853394877 z_k1_s0001
                                 - 0.110999913621278 z_k1_s0002
                                 - 0.324000064342914 z_k1_s0003
                                 - 0.295000265731792 z_k1_s0004
                                 - 0.271999489868619 z_k1_s0005
                                 - 0.0869999670328634 z_k1_s0006
                                 - 0.141999910081257 z_k1_s0007
                                 - 0.0780000441501185 z_k1_s0010
                                 - 0.219000408059986 z_k1_s0011
                                 - 0.0510000288673852 z_k1_s0012
                                 - 0.204999958202214 z_k1_s0013
                                 - 0.066999974790478 z_k1_s0014
                                 - 0.213000215263798 z_k1_s0015
                                 - 0.068000006923176 z_k1_s0016
                                 - 0.0459999629039076 z_k1_s0017
                                 - 0.296998908651708 z_k1_s0020
                                 - 0.107999697391266 z_k1_s0021
                                 - 0.349999307294964 z_k1_s0022
                                 - 0.0909997902767736 z_k1_s0023
                                 - 0.174998913149555 z_k1_s0024
                                 - 0.264999031224574 z_k1_s0025
                                 - 0.11299968749548 z_k1_s0026
                                 - 0.0819997784692717 z_k1_s0027
                                 - 0.112000238556708 z_k1_s0030
                                 - 0.101999661560962 z_k1_s0031
                                 - 0.0509999825803749 z_k1_s0032
                                 - 0.0849995914675574 z_k1_s0033
                                 - 0.0599999705769179 z_k1_s0034
                                 - 0.103000022860324 z_k1_s0035
                                 - 0.127999947350751 z_k1_s0036
                                 - 0.118000129354474 z_k1_s0037
                                 - 0.274001148082744 z_k1_s0040
                                 - 0.181001337648475 z_k1_s0041
                                 - 0.0890003669545873 z_k1_s0042
                                 - 0.100000441155844 z_k1_s0043
                                 - 0.114000535593808 z_k1_s0044
                                 - 0.15100113528141 z_k1_s0045
                                 - 0.143000147712709 z_k1_s0046
                                 - 0.115000542339377 z_k1_s0047
                                 - 0.0870000088066668 z_k1_s0050
                                 - 0.112999946096046 z_k1_s0051
                                 - 0.345000012647137 z_k1_s0052
                                 - 0.0809999636476074 z_k1_s0053
                                 - 0.31999982448439 z_k1_s0054
                                 - 0.103000000030886 z_k1_s0055
                                 - 0.160999919768705 z_k1_s0056
                                 - 0.0819999711741173 z_k1_s0057
                                 - 0.55500133910297 z_k1_s0060
                                 - 0.300001498651376 z_k1_s0061
                                 - 0.15500021686907 z_k1_s0062
                                 - 0.321999442522672 z_k1_s0063
                                 - 0.163000272759797 z_k1_s0064
                                 - 0.43100011444619 z_k1_s0065
                                 - 0.259001421249206 z_k1_s0066
                                 - 0.204000350161967 z_k1_s0067
                                 - 0.167999212454634 z_k1_s0070
                                 - 0.107999885663508 z_k1_s0071
                                 - 0.0969999320718377 z_k1_s0072
                                 - 0.126999884267264 z_k1_s0073
                                 - 0.142000004764909 z_k1_s0074
                                 - 0.222999702412642 z_k1_s0075
                                 - 0.143999943817965 z_k1_s0076
                                 - 0.435000462033136 z_k1_s0077
                                 - 0.491000933259302 z_k1_s0100
                                 - 0.37300014419367 z_k1_s0101
                                 - 0.220000531116847 z_k1_s0102
                                 - 0.204000393304488 z_k1_s0103
                                 - 0.35800058324587 z_k1_s0104
                                 - 0.143000436146156 z_k1_s0105
                                 - 0.164000276074106 z_k1_s0106
                                 - 0.320000483242032 z_k1_s0107
                                 - 0.0939999741500141 z_k1_s0110
                                 - 0.0729999221080314 z_k1_s0111
                                 - 0.166999637458232 z_k1_s0112
                                 - 0.156999612676336 z_k1_s0113
                                 - 0.181999674631077 z_k1_s0114
                                 - 0.318000270464683 z_k1_s0115
                                 - 0.190999438134971 z_k1_s0116
                                 - 0.111999889357521 z_k1_s0117
                                 - 0.0339999752930508 z_k1_s0120
                                 - 0.0920001919929245 z_k1_s0121
                                 - 0.093000124851553 z_k1_s0122
                                 - 0.115999738599846 z_k1_s0123
                                 - 0.0370000054689037 z_k1_s0124
                                 - 0.0799998396895458 z_k1_s0125
                                 - 0.0940000577101816 z_k1_s0126
                                 - 0.0329999845344305 z_k1_s0127
                                 - 0.337998850535231 z_k1_s0130
                                 - 0.18899924430852 z_k1_s0131
                                 - 0.269998702298578 z_k1_s0132
                                 - 0.285998557531301 z_k1_s0133
                                 - 0.108999662745871 z_k1_s0134
                                 - 0.0879996618785266 z_k1_s0135
                                 - 0.141999555037758 z_k1_s0136
                                 - 0.10499969893769 z_k1_s0137
                                 - 0.216999535069877 z_k1_s0140
                                 - 0.103999815814091 z_k1_s0141
                                 - 0.444999061723925 z_k1_s0142
                                 - 0.311999611842088 z_k1_s0143
                                 - 0.138999878709077 z_k1_s0144
                                 - 0.179999712832045 z_k1_s0145
                                 - 0.137999834637839 z_k1_s0146
                                 - 0.108999871770465 z_k1_s0147
                                 - 0.0959999506439787 z_k1_s0150
                                 - 0.118999960569093 z_k1_s0151
                                 - 0.104999924266857 z_k1_s0152
                                 - 0.265999993742656 z_k1_s0153
                                 - 0.199999723175002 z_k1_s0154
                                 - 0.210999304269733 z_k1_s0155
                                 - 0.253999564912274 z_k1_s0156
                                 - 0.140999818758372 z_k1_s0157
                                 - 0.141999493372627 z_k1_s0160
                                 - 0.0779999669298506 z_k1_s0161
                                 - 0.0769999041807974 z_k1_s0162
                                 - 0.251000091524114 z_k1_s0163
                                 - 0.193000175076232 z_k1_s0164
                                 - 0.119999864892141 z_k1_s0165
                                 - 0.156999887108837 z_k1_s0166
                                 - 0.0769999041807974 z_k1_s0167
                                 - 0.0380000233012368 z_k1_s0170
                                 - 0.0470000433225919 z_k1_s0171
                                 - 0.15599997963658 z_k1_s0172
                                 - 0.0440000533488178 z_k1_s0173
                                 - 0.100000166792798 z_k1_s0174
                                 - 0.0470000433225919 z_k1_s0175
                                 - 0.034000036669538 z_k1_s0176
                                 - 0.0350000333274627 z_k1_s0177
                                 - 0.433999218384978 z_k1_s1000
                                 - 0.515999221688356 z_k1_s1001
                                 - 0.263999814135572 z_k1_s1002
                                 - 0.965000693874096 z_k1_s1003
                                 - 1.05599603394731 z_k1_s1004
                                 - 0.620000417976146 z_k1_s1005
                                 - 0.387999360631639 z_k1_s1006
                                 - 1.12799954764236 z_k1_s1007
                                 - 0.155999852840413 z_k1_s1010
                                 - 0.460000133759962 z_k1_s1011
                                 - 0.204999835001813 z_k1_s1012
                                 - 0.317999151251833 z_k1_s1013
                                 - 0.163000026006256 z_k1_s1014
                                 - 0.458000154569693 z_k1_s1015
                                 - 0.378999746554551 z_k1_s1016
                                 - 0.321000350036739 z_k1_s1017
                                 - 0.145999513836409 z_k1_s1020
                                 - 0.417997908624424 z_k1_s1021
                                 - 0.584996605554908 z_k1_s1022
                                 - 0.307998840543017 z_k1_s1023
                                 - 0.656997497064913 z_k1_s1024
                                 - 0.277999152011529 z_k1_s1025
                                 - 0.340997962089473 z_k1_s1026
                                 - 0.418997089231554 z_k1_s1027
                                 - 0.263000458837251 z_k1_s1030
                                 - 0.122999922289089 z_k1_s1031
                                 - 0.266999831310462 z_k1_s1032
                                 - 0.356000400080221 z_k1_s1033
                                 - 0.461000333741638 z_k1_s1034
                                 - 0.197999874904387 z_k1_s1035
                                 - 0.108999868634273 z_k1_s1036
                                 - 0.0980000005839502 z_k1_s1037
                                 - 0.567001590725584 z_k1_s1040
                                 - 0.239000620189345 z_k1_s1041
                                 - 0.332002250374531 z_k1_s1042
                                 - 0.520002378656659 z_k1_s1043
                                 - 0.362001189140836 z_k1_s1044
                                 - 0.293000677972551 z_k1_s1045
                                 - 0.524000706489168 z_k1_s1046
                                 - 0.443000947815002 z_k1_s1047
                                 - 0.519999319590992 z_k1_s1050
                                 - 0.23799942538985 z_k1_s1051
                                 - 0.358999574121452 z_k1_s1052
                                 - 0.125999870223954 z_k1_s1053
                                 - 0.249000683092124 z_k1_s1054
                                 - 0.267000045424415 z_k1_s1055
                                 - 0.126999860142403 z_k1_s1056
                                 - 0.395000579784937 z_k1_s1057
                                 - 0.935998336906156 z_k1_s1060
                                 - 0.788997803243469 z_k1_s1061
                                 - 0.802998211924701 z_k1_s1062
                                 - 0.715998034113773 z_k1_s1063
                                 - 0.921997928224923 z_k1_s1064
                                 - 1.78999320653931 z_k1_s1065
                                 - 0.970995601118984 z_k1_s1066
                                 - 0.587998377160495 z_k1_s1067
                                 - 0.786000437558296 z_k1_s1070
                                 - 0.187000000999527 z_k1_s1071
                                 - 0.430998453640067 z_k1_s1072
                                 - 0.562998934733691 z_k1_s1073
                                 - 0.383000848605167 z_k1_s1074
                                 - 0.168000098319329 z_k1_s1075
                                 - 0.340000572908041 z_k1_s1076
                                 - 0.28299992256776 z_k1_s1077
                                 - 0.401998706927081 z_k1_s1100
                                 - 0.417998880908563 z_k1_s1101
                                 - 0.550998501947721 z_k1_s1102
                                 - 0.419998776781634 z_k1_s1103
                                 - 1.26099678177989 z_k1_s1104
                                 - 0.593997270215659 z_k1_s1105
                                 - 0.36899891452604 z_k1_s1106
                                 - 1.01999724511188 z_k1_s1107
                                 - 0.696998933998017 z_k1_s1110
                                 - 1.02299404813501 z_k1_s1111
                                 - 1.29800025654274 z_k1_s1112
                                 - 1.01100013219175 z_k1_s1113
                                 - 0.712998801906365 z_k1_s1114
                                 - 0.601998820544 z_k1_s1115
                                 - 1.07899757580623 z_k1_s1116
                                 - 1.55699861705813 z_k1_s1117
                                 - 0.986996888970139 z_k1_s1120
                                 - 0.462999597794702 z_k1_s1121
                                 - 1.43599861141603 z_k1_s1122
                                 - 1.01300095942834 z_k1_s1123
                                 - 0.93500295404688 z_k1_s1124
                                 - 0.309999630986782 z_k1_s1125
                                 - 0.593999453313305 z_k1_s1126
                                 - 1.36499705766017 z_k1_s1127
                                 - 1.27199663747878 z_k1_s1130
                                 - 0.725000156235107 z_k1_s1131
                                 - 0.633999991928602 z_k1_s1132
                                 - 1.25599681818977 z_k1_s1133
                                 - 1.20499693570594 z_k1_s1134
                                 - 0.313000041142048 z_k1_s1135
                                 - 0.475000228843717 z_k1_s1136
                                 - 1.45599822730318 z_k1_s1137
                                 - 0.432999658807823 z_k1_s1140
                                 - 0.576000913297461 z_k1_s1141
                                 - 0.708001122594796 z_k1_s1142
                                 - 0.431001711138369 z_k1_s1143
                                 - 0.334000118485337 z_k1_s1144
                                 - 0.737999114660655 z_k1_s1145
                                 - 0.427001704796025 z_k1_s1146
                                 - 0.464000735711844 z_k1_s1147
                                 - 1.22800036047355 z_k1_s1150
                                 - 1.97799673569005 z_k1_s1151
                                 - 0.657998257082076 z_k1_s1152
                                 - 0.721997806068931 z_k1_s1153
                                 - 0.442998941940966 z_k1_s1154
                                 - 0.485998804969188 z_k1_s1155
                                 - 1.2839990169646 z_k1_s1156
                                 - 0.791997265329681 z_k1_s1157
                                 - 0.57000133982929 z_k1_s1160
                                 - 1.19000532140802 z_k1_s1161
                                 - 0.803004249179929 z_k1_s1162
                                 - 0.981001842041639 z_k1_s1163
                                 - 0.735001469966229 z_k1_s1164
                                 - 0.843998912532311 z_k1_s1165
                                 - 1.09400458826528 z_k1_s1166
                                 - 1.31600251715075 z_k1_s1167
                                 - 0.254000254513968 z_k1_s1170
                                 - 0.379999285460627 z_k1_s1171
                                 - 0.696999509048427 z_k1_s1172
                                 - 0.197000210494706 z_k1_s1173
                                 - 0.301000221842054 z_k1_s1174
                                 - 0.554000438446871 z_k1_s1175
                                 - 0.349000227079292 z_k1_s1176
                                 - 0.29200018306003 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.33875 w_k1_ADM_s000
                                 + 0.924684 w_k1_AXP_s000
                                 + 1.00803 w_k1_DIS_s000
                                 + 1.10989 w_k1_FDX_s000
                                 + 0.648015 w_k1_IBM_s000
                                 + 0.932344 w_k1_INTC_s000
                                 + 0.797784 w_k1_KO_s000
                                 + 0.638059 w_k1_MAT_s000
                                 + 1.08165 w_k1_MSFT_s000
                                 + 1.03853 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.921288 w_k1_ADM_s000
                                 + 0.682121 w_k1_AXP_s000
                                 + 0.791773 w_k1_DIS_s000
                                 + 0.74083 w_k1_FDX_s000
                                 + 0.845582 w_k1_IBM_s000
                                 + 0.940191 w_k1_INTC_s000
                                 + 0.877921 w_k1_KO_s000
                                 + 0.591943 w_k1_MAT_s000
                                 + 0.748492 w_k1_MSFT_s000
                                 + 0.902668 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.711593 w_k1_ADM_s000
                                 + 0.537486 w_k1_AXP_s000
                                 + 0.56546 w_k1_DIS_s000
                                 + 0.750545 w_k1_FDX_s000
                                 + 0.622057 w_k1_IBM_s000
                                 + 0.813547 w_k1_INTC_s000
                                 + 0.698732 w_k1_KO_s000
                                 + 0.497823 w_k1_MAT_s000
                                 + 0.822237 w_k1_MSFT_s000
                                 + 0.88468 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.14948 w_k1_ADM_s000
                                 + 1.25831 w_k1_AXP_s000
                                 + 1.38335 w_k1_DIS_s000
                                 + 1.17115 w_k1_FDX_s000
                                 + 1.02757 w_k1_IBM_s000
                                 + 1.20362 w_k1_INTC_s000
                                 + 1.18567 w_k1_KO_s000 + 1.16864 w_k1_MAT_s000
                                 + 1.21284 w_k1_MSFT_s000
                                 + 1.14243 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.17037 w_k1_ADM_s000
                                 + 0.988523 w_k1_AXP_s000
                                 + 1.25306 w_k1_DIS_s000
                                 + 0.946725 w_k1_FDX_s000
                                 + 1.42592 w_k1_IBM_s000
                                 + 1.06337 w_k1_INTC_s000
                                 + 1.12554 w_k1_KO_s000 + 0.83312 w_k1_MAT_s000
                                 + 1.00675 w_k1_MSFT_s000
                                 + 1.05074 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.870594 w_k1_ADM_s000
                                 + 0.837494 w_k1_AXP_s000
                                 + 0.803668 w_k1_DIS_s000
                                 + 0.927422 w_k1_FDX_s000
                                 + 0.831211 w_k1_IBM_s000
                                 + 1.1118 w_k1_INTC_s000
                                 + 0.994181 w_k1_KO_s000
                                 + 0.71516 w_k1_MAT_s000
                                 + 0.994379 w_k1_MSFT_s000
                                 + 1.08509 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.801893 w_k1_ADM_s000
                                 + 0.589934 w_k1_AXP_s000
                                 + 0.747603 w_k1_DIS_s000
                                 + 0.65944 w_k1_FDX_s000
                                 + 1.12778 w_k1_IBM_s000
                                 + 0.934691 w_k1_INTC_s000
                                 + 0.895966 w_k1_KO_s000
                                 + 0.555854 w_k1_MAT_s000
                                 + 0.798424 w_k1_MSFT_s000
                                 + 1.00516 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.24413 w_k1_ADM_s000
                                 + 1.38604 w_k1_AXP_s000
                                 + 1.75622 w_k1_DIS_s000
                                 + 1.34649 w_k1_FDX_s000
                                 + 1.28841 w_k1_IBM_s000
                                 + 1.61034 w_k1_INTC_s000
                                 + 1.47486 w_k1_KO_s000
                                 + 0.746804 w_k1_MAT_s000
                                 + 1.31688 w_k1_MSFT_s000
                                 + 1.24339 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.06521 w_k1_ADM_s001
                                 + 0.859233 w_k1_AXP_s001
                                 + 1.27651 w_k1_DIS_s001
                                 + 1.12694 w_k1_FDX_s001
                                 + 0.584376 w_k1_IBM_s001
                                 + 0.997634 w_k1_INTC_s001
                                 + 0.854309 w_k1_KO_s001
                                 + 0.601918 w_k1_MAT_s001
                                 + 1.20941 w_k1_MSFT_s001
                                 + 0.853855 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.781734 w_k1_ADM_s001
                                 + 1.01675 w_k1_AXP_s001
                                 + 1.03745 w_k1_DIS_s001
                                 + 0.830777 w_k1_FDX_s001
                                 + 1.26204 w_k1_IBM_s001
                                 + 0.862568 w_k1_INTC_s001
                                 + 1.14238 w_k1_KO_s001 + 1.71709 w_k1_MAT_s001
                                 + 0.958603 w_k1_MSFT_s001
                                 + 0.859929 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.19624 w_k1_ADM_s001
                                 + 1.22843 w_k1_AXP_s001
                                 + 1.48241 w_k1_DIS_s001
                                 + 1.40529 w_k1_FDX_s001
                                 + 0.846302 w_k1_IBM_s001
                                 + 1.13603 w_k1_INTC_s001
                                 + 1.06016 w_k1_KO_s001
                                 + 0.836836 w_k1_MAT_s001
                                 + 1.27348 w_k1_MSFT_s001
                                 + 1.06277 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.917291 w_k1_ADM_s001
                                 + 1.16726 w_k1_AXP_s001
                                 + 1.42201 w_k1_DIS_s001
                                 + 1.07664 w_k1_FDX_s001
                                 + 1.07409 w_k1_IBM_s001
                                 + 1.06079 w_k1_INTC_s001
                                 + 1.39091 w_k1_KO_s001 + 1.63127 w_k1_MAT_s001
                                 + 1.34443 w_k1_MSFT_s001
                                 + 1.00722 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.835026 w_k1_ADM_s001
                                 + 0.731128 w_k1_AXP_s001
                                 + 0.618027 w_k1_DIS_s001
                                 + 0.747527 w_k1_FDX_s001
                                 + 0.660743 w_k1_IBM_s001
                                 + 0.666927 w_k1_INTC_s001
                                 + 0.632376 w_k1_KO_s001
                                 + 0.439948 w_k1_MAT_s001
                                 + 0.697681 w_k1_MSFT_s001
                                 + 0.663058 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.785908 w_k1_ADM_s001
                                 + 0.848816 w_k1_AXP_s001
                                 + 0.979199 w_k1_DIS_s001
                                 + 0.904049 w_k1_FDX_s001
                                 + 0.856685 w_k1_IBM_s001
                                 + 0.886679 w_k1_INTC_s001
                                 + 0.935569 w_k1_KO_s001
                                 + 0.914966 w_k1_MAT_s001
                                 + 0.947485 w_k1_MSFT_s001
                                 + 0.954874 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.33572 w_k1_ADM_s001
                                 + 1.59948 w_k1_AXP_s001
                                 + 1.74041 w_k1_DIS_s001 + 1.3556 w_k1_FDX_s001
                                 + 1.20878 w_k1_IBM_s001
                                 + 1.13325 w_k1_INTC_s001
                                 + 1.59011 w_k1_KO_s001 + 1.08232 w_k1_MAT_s001
                                 + 1.42098 w_k1_MSFT_s001
                                 + 1.26637 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.15666 w_k1_ADM_s001
                                 + 1.2285 w_k1_AXP_s001 + 1.07455 w_k1_DIS_s001
                                 + 1.15298 w_k1_FDX_s001
                                 + 0.899933 w_k1_IBM_s001
                                 + 0.987243 w_k1_INTC_s001
                                 + 0.948391 w_k1_KO_s001
                                 + 0.828945 w_k1_MAT_s001
                                 + 1.00727 w_k1_MSFT_s001
                                 + 0.957698 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.78916 w_k1_ADM_s002
                                 + 0.700635 w_k1_AXP_s002
                                 + 0.602455 w_k1_DIS_s002
                                 + 0.697798 w_k1_FDX_s002
                                 + 0.925771 w_k1_IBM_s002
                                 + 0.755251 w_k1_INTC_s002
                                 + 0.815036 w_k1_KO_s002
                                 + 0.911145 w_k1_MAT_s002
                                 + 0.733415 w_k1_MSFT_s002
                                 + 0.806435 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.11708 w_k1_ADM_s002
                                 + 1.07462 w_k1_AXP_s002
                                 + 1.17063 w_k1_DIS_s002
                                 + 1.15838 w_k1_FDX_s002 + 1.3761 w_k1_IBM_s002
                                 + 1.38268 w_k1_INTC_s002
                                 + 1.30534 w_k1_KO_s002 + 1.14232 w_k1_MAT_s002
                                 + 1.06837 w_k1_MSFT_s002
                                 + 1.07072 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.831667 w_k1_ADM_s002
                                 + 0.769783 w_k1_AXP_s002
                                 + 0.831601 w_k1_DIS_s002
                                 + 0.987113 w_k1_FDX_s002
                                 + 0.720003 w_k1_IBM_s002
                                 + 1.12344 w_k1_INTC_s002
                                 + 0.80995 w_k1_KO_s002
                                 + 0.619238 w_k1_MAT_s002
                                 + 0.990072 w_k1_MSFT_s002
                                 + 0.834592 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.18511 w_k1_ADM_s002
                                 + 1.39578 w_k1_AXP_s002
                                 + 1.29141 w_k1_DIS_s002
                                 + 1.27944 w_k1_FDX_s002
                                 + 1.04578 w_k1_IBM_s002
                                 + 1.20629 w_k1_INTC_s002
                                 + 1.06906 w_k1_KO_s002 + 1.30392 w_k1_MAT_s002
                                 + 1.15066 w_k1_MSFT_s002
                                 + 1.13987 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.946179 w_k1_ADM_s002
                                 + 0.919362 w_k1_AXP_s002
                                 + 1.20211 w_k1_DIS_s002 + 1.0747 w_k1_FDX_s002
                                 + 0.941245 w_k1_IBM_s002
                                 + 1.09935 w_k1_INTC_s002
                                 + 1.12598 w_k1_KO_s002
                                 + 0.910377 w_k1_MAT_s002
                                 + 1.14772 w_k1_MSFT_s002
                                 + 0.913791 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.844082 w_k1_ADM_s002
                                 + 1.06165 w_k1_AXP_s002
                                 + 0.964988 w_k1_DIS_s002
                                 + 0.872894 w_k1_FDX_s002
                                 + 1.20997 w_k1_IBM_s002
                                 + 1.02368 w_k1_INTC_s002
                                 + 1.08184 w_k1_KO_s002 + 1.38754 w_k1_MAT_s002
                                 + 0.952371 w_k1_MSFT_s002
                                 + 1.01518 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.01987 w_k1_ADM_s002
                                 + 1.42772 w_k1_AXP_s002
                                 + 1.95551 w_k1_DIS_s002
                                 + 1.46892 w_k1_FDX_s002
                                 + 1.41761 w_k1_IBM_s002
                                 + 1.56179 w_k1_INTC_s002
                                 + 1.72386 w_k1_KO_s002 + 1.3678 w_k1_MAT_s002
                                 + 1.51735 w_k1_MSFT_s002
                                 + 1.29937 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.906905 w_k1_ADM_s002
                                 + 1.38495 w_k1_AXP_s002
                                 + 1.19082 w_k1_DIS_s002
                                 + 1.08438 w_k1_FDX_s002
                                 + 1.82071 w_k1_IBM_s002
                                 + 1.17492 w_k1_INTC_s002 + 1.4452 w_k1_KO_s002
                                 + 2.53442 w_k1_MAT_s002
                                 + 1.13793 w_k1_MSFT_s002
                                 + 1.18114 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.807849 w_k1_ADM_s003
                                 + 1.01991 w_k1_AXP_s003
                                 + 0.878301 w_k1_DIS_s003
                                 + 0.8463 w_k1_FDX_s003 + 1.44161 w_k1_IBM_s003
                                 + 0.755495 w_k1_INTC_s003
                                 + 1.19228 w_k1_KO_s003 + 1.86761 w_k1_MAT_s003
                                 + 0.974965 w_k1_MSFT_s003
                                 + 1.0562 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.1865 w_k1_ADM_s003
                                 + 1.13487 w_k1_AXP_s003
                                 + 1.63644 w_k1_DIS_s003
                                 + 1.44566 w_k1_FDX_s003 + 1.0423 w_k1_IBM_s003
                                 + 1.01565 w_k1_INTC_s003
                                 + 1.15801 w_k1_KO_s003 + 2.0703 w_k1_MAT_s003
                                 + 1.29472 w_k1_MSFT_s003
                                 + 1.21731 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06927 w_k1_ADM_s003
                                 + 1.06559 w_k1_AXP_s003
                                 + 0.897746 w_k1_DIS_s003
                                 + 1.05249 w_k1_FDX_s003
                                 + 1.02363 w_k1_IBM_s003
                                 + 0.74659 w_k1_INTC_s003
                                 + 0.912036 w_k1_KO_s003
                                 + 1.22738 w_k1_MAT_s003
                                 + 0.914218 w_k1_MSFT_s003
                                 + 0.918591 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.898348 w_k1_ADM_s003
                                 + 0.718312 w_k1_AXP_s003
                                 + 0.60922 w_k1_DIS_s003
                                 + 0.666788 w_k1_FDX_s003
                                 + 0.958037 w_k1_IBM_s003
                                 + 0.677088 w_k1_INTC_s003
                                 + 0.854605 w_k1_KO_s003
                                 + 0.930179 w_k1_MAT_s003
                                 + 0.703254 w_k1_MSFT_s003
                                 + 0.851662 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06236 w_k1_ADM_s003
                                 + 0.821324 w_k1_AXP_s003
                                 + 0.7807 w_k1_DIS_s003
                                 + 0.892705 w_k1_FDX_s003
                                 + 0.88109 w_k1_IBM_s003
                                 + 0.727959 w_k1_INTC_s003
                                 + 0.868688 w_k1_KO_s003
                                 + 0.624729 w_k1_MAT_s003
                                 + 0.861643 w_k1_MSFT_s003
                                 + 0.845185 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.812142 w_k1_ADM_s003
                                 + 0.902117 w_k1_AXP_s003
                                 + 1.07611 w_k1_DIS_s003
                                 + 1.00316 w_k1_FDX_s003
                                 + 0.980438 w_k1_IBM_s003
                                 + 0.992351 w_k1_INTC_s003
                                 + 1.21725 w_k1_KO_s003 + 1.21339 w_k1_MAT_s003
                                 + 1.09782 w_k1_MSFT_s003
                                 + 1.10186 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.07491 w_k1_ADM_s003
                                 + 1.1265 w_k1_AXP_s003 + 1.23338 w_k1_DIS_s003
                                 + 1.14057 w_k1_FDX_s003
                                 + 1.34391 w_k1_IBM_s003
                                 + 1.02689 w_k1_INTC_s003
                                 + 1.19856 w_k1_KO_s003 + 1.98495 w_k1_MAT_s003
                                 + 1.06394 w_k1_MSFT_s003
                                 + 1.07084 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.12552 w_k1_ADM_s003
                                 + 0.821692 w_k1_AXP_s003
                                 + 0.867117 w_k1_DIS_s003
                                 + 0.985472 w_k1_FDX_s003
                                 + 0.728826 w_k1_IBM_s003
                                 + 0.832356 w_k1_INTC_s003
                                 + 0.761698 w_k1_KO_s003
                                 + 0.658084 w_k1_MAT_s003
                                 + 0.948199 w_k1_MSFT_s003
                                 + 0.884179 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07181 w_k1_ADM_s004
                                 + 0.816767 w_k1_AXP_s004
                                 + 1.18609 w_k1_DIS_s004
                                 + 1.14133 w_k1_FDX_s004
                                 + 0.966497 w_k1_IBM_s004
                                 + 0.973888 w_k1_INTC_s004
                                 + 0.940833 w_k1_KO_s004
                                 + 0.716127 w_k1_MAT_s004
                                 + 1.03503 w_k1_MSFT_s004
                                 + 1.09289 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.01053 w_k1_ADM_s004
                                 + 0.805545 w_k1_AXP_s004
                                 + 0.730946 w_k1_DIS_s004
                                 + 0.93235 w_k1_FDX_s004
                                 + 0.783498 w_k1_IBM_s004
                                 + 0.785737 w_k1_INTC_s004
                                 + 0.732878 w_k1_KO_s004
                                 + 0.603141 w_k1_MAT_s004
                                 + 0.826131 w_k1_MSFT_s004
                                 + 0.913987 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.47494 w_k1_ADM_s004
                                 + 1.51684 w_k1_AXP_s004
                                 + 1.89951 w_k1_DIS_s004
                                 + 1.48189 w_k1_FDX_s004
                                 + 1.35919 w_k1_IBM_s004
                                 + 1.13774 w_k1_INTC_s004
                                 + 1.35588 w_k1_KO_s004 + 1.56879 w_k1_MAT_s004
                                 + 1.43237 w_k1_MSFT_s004
                                 + 1.29288 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.28 w_k1_ADM_s004
                                 + 1.42641 w_k1_AXP_s004
                                 + 1.62264 w_k1_DIS_s004
                                 + 1.10748 w_k1_FDX_s004 + 1.8451 w_k1_IBM_s004
                                 + 1.21411 w_k1_INTC_s004
                                 + 1.52876 w_k1_KO_s004 + 1.98053 w_k1_MAT_s004
                                 + 1.37631 w_k1_MSFT_s004
                                 + 1.14644 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.395 w_k1_ADM_s004
                                 + 1.44418 w_k1_AXP_s004
                                 + 1.23912 w_k1_DIS_s004
                                 + 1.26422 w_k1_FDX_s004
                                 + 1.01113 w_k1_IBM_s004
                                 + 0.986118 w_k1_INTC_s004
                                 + 0.930804 w_k1_KO_s004
                                 + 1.08542 w_k1_MAT_s004
                                 + 1.03339 w_k1_MSFT_s004
                                 + 1.13286 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.890367 w_k1_ADM_s004
                                 + 0.793823 w_k1_AXP_s004
                                 + 0.905191 w_k1_DIS_s004
                                 + 0.835251 w_k1_FDX_s004
                                 + 0.931347 w_k1_IBM_s004
                                 + 0.835516 w_k1_INTC_s004
                                 + 0.771616 w_k1_KO_s004
                                 + 0.90643 w_k1_MAT_s004
                                 + 0.911216 w_k1_MSFT_s004
                                 + 0.867588 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.10235 w_k1_ADM_s004
                                 + 1.08327 w_k1_AXP_s004
                                 + 0.953721 w_k1_DIS_s004
                                 + 0.89207 w_k1_FDX_s004
                                 + 1.30532 w_k1_IBM_s004
                                 + 0.993629 w_k1_INTC_s004
                                 + 0.975658 w_k1_KO_s004
                                 + 1.07564 w_k1_MAT_s004
                                 + 0.98189 w_k1_MSFT_s004
                                 + 1.04569 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.85346 w_k1_ADM_s004
                                 + 0.642625 w_k1_AXP_s004
                                 + 0.639848 w_k1_DIS_s004
                                 + 0.683421 w_k1_FDX_s004
                                 + 0.869553 w_k1_IBM_s004
                                 + 0.685878 w_k1_INTC_s004
                                 + 0.705976 w_k1_KO_s004
                                 + 0.572874 w_k1_MAT_s004
                                 + 0.752909 w_k1_MSFT_s004
                                 + 0.748058 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.42714 w_k1_ADM_s005
                                 + 1.07601 w_k1_AXP_s005 + 1.3415 w_k1_DIS_s005
                                 + 1.23743 w_k1_FDX_s005
                                 + 0.952273 w_k1_IBM_s005
                                 + 0.986579 w_k1_INTC_s005
                                 + 1.18659 w_k1_KO_s005 + 1.62053 w_k1_MAT_s005
                                 + 1.21696 w_k1_MSFT_s005 + 1.2113 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.37399 w_k1_ADM_s005
                                 + 1.3721 w_k1_AXP_s005 + 1.36377 w_k1_DIS_s005
                                 + 1.19894 w_k1_FDX_s005
                                 + 1.33298 w_k1_IBM_s005
                                 + 0.930091 w_k1_INTC_s005
                                 + 1.23104 w_k1_KO_s005 + 2.30503 w_k1_MAT_s005
                                 + 1.15221 w_k1_MSFT_s005
                                 + 1.24075 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.15172 w_k1_ADM_s005
                                 + 0.781216 w_k1_AXP_s005
                                 + 0.85798 w_k1_DIS_s005
                                 + 0.842153 w_k1_FDX_s005
                                 + 1.09902 w_k1_IBM_s005
                                 + 0.969665 w_k1_INTC_s005
                                 + 1.10589 w_k1_KO_s005 + 1.22817 w_k1_MAT_s005
                                 + 0.917215 w_k1_MSFT_s005
                                 + 0.956569 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.36298 w_k1_ADM_s005
                                 + 1.08843 w_k1_AXP_s005
                                 + 1.95616 w_k1_DIS_s005
                                 + 1.36453 w_k1_FDX_s005
                                 + 1.38656 w_k1_IBM_s005
                                 + 1.41848 w_k1_INTC_s005 + 1.4632 w_k1_KO_s005
                                 + 0.991449 w_k1_MAT_s005
                                 + 1.41047 w_k1_MSFT_s005
                                 + 1.39813 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.00972 w_k1_ADM_s005
                                 + 1.08993 w_k1_AXP_s005
                                 + 1.19395 w_k1_DIS_s005
                                 + 1.04561 w_k1_FDX_s005
                                 + 1.16725 w_k1_IBM_s005
                                 + 1.0241 w_k1_INTC_s005 + 1.1685 w_k1_KO_s005
                                 + 1.26595 w_k1_MAT_s005
                                 + 1.00612 w_k1_MSFT_s005
                                 + 1.20199 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.726885 w_k1_ADM_s005
                                 + 0.537764 w_k1_AXP_s005
                                 + 0.581914 w_k1_DIS_s005
                                 + 0.650199 w_k1_FDX_s005
                                 + 0.761812 w_k1_IBM_s005
                                 + 0.550121 w_k1_INTC_s005
                                 + 0.644302 w_k1_KO_s005
                                 + 0.649689 w_k1_MAT_s005
                                 + 0.680791 w_k1_MSFT_s005
                                 + 0.864501 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.984079 w_k1_ADM_s005
                                 + 0.71868 w_k1_AXP_s005
                                 + 0.773721 w_k1_DIS_s005
                                 + 0.895498 w_k1_FDX_s005
                                 + 0.60535 w_k1_IBM_s005
                                 + 0.884454 w_k1_INTC_s005
                                 + 0.761767 w_k1_KO_s005
                                 + 0.938893 w_k1_MAT_s005
                                 + 0.923631 w_k1_MSFT_s005
                                 + 0.913218 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.16814 w_k1_ADM_s005
                                 + 0.943845 w_k1_AXP_s005
                                 + 0.797615 w_k1_DIS_s005
                                 + 1.08852 w_k1_FDX_s005
                                 + 0.742725 w_k1_IBM_s005
                                 + 0.876625 w_k1_INTC_s005
                                 + 0.851985 w_k1_KO_s005
                                 + 1.05927 w_k1_MAT_s005
                                 + 1.08315 w_k1_MSFT_s005
                                 + 1.00578 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00067 w_k1_ADM_s006
                                 + 0.8677 w_k1_AXP_s006
                                 + 0.868018 w_k1_DIS_s006
                                 + 0.873234 w_k1_FDX_s006
                                 + 0.886468 w_k1_IBM_s006
                                 + 0.985853 w_k1_INTC_s006
                                 + 0.876091 w_k1_KO_s006
                                 + 0.860725 w_k1_MAT_s006
                                 + 1.01641 w_k1_MSFT_s006
                                 + 0.86282 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.00147 w_k1_ADM_s006
                                 + 0.692811 w_k1_AXP_s006
                                 + 0.556861 w_k1_DIS_s006
                                 + 0.720686 w_k1_FDX_s006
                                 + 0.762643 w_k1_IBM_s006
                                 + 0.916582 w_k1_INTC_s006
                                 + 0.765736 w_k1_KO_s006
                                 + 0.566415 w_k1_MAT_s006
                                 + 0.769253 w_k1_MSFT_s006
                                 + 0.823434 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.803933 w_k1_ADM_s006
                                 + 0.907701 w_k1_AXP_s006
                                 + 0.87868 w_k1_DIS_s006
                                 + 0.844981 w_k1_FDX_s006
                                 + 1.27468 w_k1_IBM_s006
                                 + 1.14637 w_k1_INTC_s006
                                 + 1.03318 w_k1_KO_s006 + 1.36182 w_k1_MAT_s006
                                 + 1.00815 w_k1_MSFT_s006
                                 + 0.986526 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.38819 w_k1_ADM_s006
                                 + 1.53687 w_k1_AXP_s006
                                 + 1.17422 w_k1_DIS_s006
                                 + 1.26917 w_k1_FDX_s006
                                 + 1.21979 w_k1_IBM_s006
                                 + 1.24079 w_k1_INTC_s006
                                 + 1.12908 w_k1_KO_s006 + 1.46415 w_k1_MAT_s006
                                 + 1.20609 w_k1_MSFT_s006
                                 + 1.23852 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.705083 w_k1_ADM_s006
                                 + 0.698485 w_k1_AXP_s006
                                 + 0.661849 w_k1_DIS_s006
                                 + 0.710731 w_k1_FDX_s006
                                 + 1.11111 w_k1_IBM_s006
                                 + 0.840612 w_k1_INTC_s006
                                 + 0.844556 w_k1_KO_s006
                                 + 0.965627 w_k1_MAT_s006
                                 + 0.991793 w_k1_MSFT_s006
                                 + 0.849531 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03611 w_k1_ADM_s006
                                 + 1.21776 w_k1_AXP_s006
                                 + 1.15229 w_k1_DIS_s006
                                 + 1.06209 w_k1_FDX_s006
                                 + 1.35537 w_k1_IBM_s006
                                 + 1.04906 w_k1_INTC_s006
                                 + 1.10544 w_k1_KO_s006 + 1.33874 w_k1_MAT_s006
                                 + 1.19058 w_k1_MSFT_s006
                                 + 1.10477 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.21584 w_k1_ADM_s006
                                 + 1.11077 w_k1_AXP_s006
                                 + 1.62198 w_k1_DIS_s006 + 1.254 w_k1_FDX_s006
                                 + 1.20329 w_k1_IBM_s006
                                 + 1.11734 w_k1_INTC_s006
                                 + 1.23934 w_k1_KO_s006 + 1.178 w_k1_MAT_s006
                                 + 1.29375 w_k1_MSFT_s006
                                 + 1.24369 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.06948 w_k1_ADM_s006
                                 + 0.76389 w_k1_AXP_s006
                                 + 1.04571 w_k1_DIS_s006
                                 + 1.02538 w_k1_FDX_s006
                                 + 0.771909 w_k1_IBM_s006
                                 + 1.00018 w_k1_INTC_s006
                                 + 0.898605 w_k1_KO_s006
                                 + 0.706041 w_k1_MAT_s006
                                 + 1.2011 w_k1_MSFT_s006
                                 + 0.885831 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.01006 w_k1_ADM_s007
                                 + 0.944791 w_k1_AXP_s007
                                 + 0.902007 w_k1_DIS_s007
                                 + 0.86911 w_k1_FDX_s007 + 1.3267 w_k1_IBM_s007
                                 + 0.943163 w_k1_INTC_s007
                                 + 1.03451 w_k1_KO_s007 + 1.50555 w_k1_MAT_s007
                                 + 1.03122 w_k1_MSFT_s007
                                 + 0.922764 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.5916 w_k1_ADM_s007
                                 + 1.25585 w_k1_AXP_s007
                                 + 1.72454 w_k1_DIS_s007
                                 + 1.27574 w_k1_FDX_s007
                                 + 1.23791 w_k1_IBM_s007
                                 + 1.07597 w_k1_INTC_s007
                                 + 1.27282 w_k1_KO_s007 + 2.07155 w_k1_MAT_s007
                                 + 1.23903 w_k1_MSFT_s007
                                 + 1.30022 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.38851 w_k1_ADM_s007
                                 + 1.62409 w_k1_AXP_s007
                                 + 1.58909 w_k1_DIS_s007
                                 + 1.46703 w_k1_FDX_s007
                                 + 1.21917 w_k1_IBM_s007
                                 + 1.48119 w_k1_INTC_s007
                                 + 1.05954 w_k1_KO_s007 + 1.33363 w_k1_MAT_s007
                                 + 1.27898 w_k1_MSFT_s007
                                 + 0.964948 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.826486 w_k1_ADM_s007
                                 + 0.991655 w_k1_AXP_s007
                                 + 0.804404 w_k1_DIS_s007
                                 + 0.825375 w_k1_FDX_s007
                                 + 1.38935 w_k1_IBM_s007
                                 + 1.08971 w_k1_INTC_s007
                                 + 1.07644 w_k1_KO_s007 + 2.01636 w_k1_MAT_s007
                                 + 0.979093 w_k1_MSFT_s007
                                 + 0.926239 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.01516 w_k1_ADM_s007
                                 + 0.915034 w_k1_AXP_s007
                                 + 1.19982 w_k1_DIS_s007
                                 + 1.08335 w_k1_FDX_s007
                                 + 0.852035 w_k1_IBM_s007
                                 + 0.913969 w_k1_INTC_s007
                                 + 0.982514 w_k1_KO_s007
                                 + 1.18992 w_k1_MAT_s007
                                 + 1.19122 w_k1_MSFT_s007
                                 + 1.11455 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.0609 w_k1_ADM_s007
                                 + 1.05036 w_k1_AXP_s007
                                 + 1.22535 w_k1_DIS_s007
                                 + 1.04987 w_k1_FDX_s007
                                 + 1.19393 w_k1_IBM_s007
                                 + 0.87179 w_k1_INTC_s007 + 1.128 w_k1_KO_s007
                                 + 2.12146 w_k1_MAT_s007
                                 + 1.14704 w_k1_MSFT_s007
                                 + 1.10644 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.25312 w_k1_ADM_s007
                                 + 1.23864 w_k1_AXP_s007
                                 + 0.943865 w_k1_DIS_s007
                                 + 1.12354 w_k1_FDX_s007
                                 + 1.01478 w_k1_IBM_s007
                                 + 0.908482 w_k1_INTC_s007
                                 + 0.951765 w_k1_KO_s007
                                 + 1.49607 w_k1_MAT_s007
                                 + 1.13253 w_k1_MSFT_s007
                                 + 0.976852 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.967961 w_k1_ADM_s007
                                 + 0.736046 w_k1_AXP_s007
                                 + 0.724356 w_k1_DIS_s007
                                 + 0.872743 w_k1_FDX_s007
                                 + 0.828985 w_k1_IBM_s007
                                 + 0.867549 w_k1_INTC_s007
                                 + 0.808672 w_k1_KO_s007
                                 + 0.86239 w_k1_MAT_s007
                                 + 0.916435 w_k1_MSFT_s007
                                 + 0.924071 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00537 w_k1_ADM_s010
                                 + 1.29519 w_k1_AXP_s010
                                 + 0.944301 w_k1_DIS_s010
                                 + 0.983977 w_k1_FDX_s010
                                 + 1.12125 w_k1_IBM_s010
                                 + 1.10714 w_k1_INTC_s010
                                 + 0.901107 w_k1_KO_s010
                                 + 1.19128 w_k1_MAT_s010
                                 + 0.988607 w_k1_MSFT_s010
                                 + 0.963534 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.990879 w_k1_ADM_s010
                                 + 0.994981 w_k1_AXP_s010
                                 + 1.34579 w_k1_DIS_s010
                                 + 1.05036 w_k1_FDX_s010
                                 + 1.26039 w_k1_IBM_s010
                                 + 1.07045 w_k1_INTC_s010
                                 + 1.33446 w_k1_KO_s010 + 1.27294 w_k1_MAT_s010
                                 + 1.33077 w_k1_MSFT_s010 + 1.0171 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.923481 w_k1_ADM_s010
                                 + 0.840012 w_k1_AXP_s010
                                 + 0.824917 w_k1_DIS_s010
                                 + 0.934956 w_k1_FDX_s010
                                 + 0.736735 w_k1_IBM_s010
                                 + 1.05458 w_k1_INTC_s010
                                 + 0.784271 w_k1_KO_s010
                                 + 0.589656 w_k1_MAT_s010
                                 + 0.932623 w_k1_MSFT_s010
                                 + 0.775873 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.692902 w_k1_ADM_s010
                                 + 0.798904 w_k1_AXP_s010
                                 + 0.557732 w_k1_DIS_s010
                                 + 0.767554 w_k1_FDX_s010
                                 + 0.761067 w_k1_IBM_s010
                                 + 1.05775 w_k1_INTC_s010
                                 + 0.710132 w_k1_KO_s010
                                 + 0.613297 w_k1_MAT_s010
                                 + 0.838708 w_k1_MSFT_s010
                                 + 0.752883 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.75201 w_k1_ADM_s010
                                 + 0.741559 w_k1_AXP_s010
                                 + 0.762924 w_k1_DIS_s010
                                 + 0.724544 w_k1_FDX_s010
                                 + 1.08261 w_k1_IBM_s010
                                 + 0.967071 w_k1_INTC_s010
                                 + 0.834202 w_k1_KO_s010
                                 + 0.842448 w_k1_MAT_s010
                                 + 0.93007 w_k1_MSFT_s010
                                 + 0.814539 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.982238 w_k1_ADM_s010
                                 + 1.65048 w_k1_AXP_s010
                                 + 1.57582 w_k1_DIS_s010
                                 + 1.31486 w_k1_FDX_s010
                                 + 1.56691 w_k1_IBM_s010
                                 + 1.46226 w_k1_INTC_s010
                                 + 1.24382 w_k1_KO_s010 + 1.80327 w_k1_MAT_s010
                                 + 1.36388 w_k1_MSFT_s010
                                 + 1.13708 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.923992 w_k1_ADM_s010
                                 + 1.18636 w_k1_AXP_s010
                                 + 1.07285 w_k1_DIS_s010
                                 + 1.09615 w_k1_FDX_s010
                                 + 0.755382 w_k1_IBM_s010
                                 + 1.2674 w_k1_INTC_s010
                                 + 0.839293 w_k1_KO_s010
                                 + 0.555478 w_k1_MAT_s010
                                 + 1.14545 w_k1_MSFT_s010
                                 + 0.94549 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.14182 w_k1_ADM_s010
                                 + 1.16053 w_k1_AXP_s010
                                 + 1.47896 w_k1_DIS_s010
                                 + 1.35123 w_k1_FDX_s010
                                 + 0.872451 w_k1_IBM_s010
                                 + 1.27914 w_k1_INTC_s010
                                 + 0.970292 w_k1_KO_s010
                                 + 0.77986 w_k1_MAT_s010
                                 + 1.41113 w_k1_MSFT_s010
                                 + 1.03127 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.984057 w_k1_ADM_s011
                                 + 1.14049 w_k1_AXP_s011 + 1.0995 w_k1_DIS_s011
                                 + 1.0598 w_k1_FDX_s011 + 1.47444 w_k1_IBM_s011
                                 + 1.06998 w_k1_INTC_s011
                                 + 1.05512 w_k1_KO_s011 + 1.19246 w_k1_MAT_s011
                                 + 1.0967 w_k1_MSFT_s011 + 1.02408 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.24921 w_k1_ADM_s011
                                 + 1.23414 w_k1_AXP_s011
                                 + 1.04281 w_k1_DIS_s011 + 1.0569 w_k1_FDX_s011
                                 + 1.99739 w_k1_IBM_s011
                                 + 1.02518 w_k1_INTC_s011
                                 + 1.25668 w_k1_KO_s011 + 2.56144 w_k1_MAT_s011
                                 + 1.10704 w_k1_MSFT_s011
                                 + 1.16683 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.844452 w_k1_ADM_s011
                                 + 0.752812 w_k1_AXP_s011
                                 + 0.867497 w_k1_DIS_s011
                                 + 0.839456 w_k1_FDX_s011
                                 + 1.39819 w_k1_IBM_s011
                                 + 1.07431 w_k1_INTC_s011
                                 + 1.10336 w_k1_KO_s011 + 1.35549 w_k1_MAT_s011
                                 + 0.98411 w_k1_MSFT_s011
                                 + 1.01556 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.57366 w_k1_ADM_s011
                                 + 1.14737 w_k1_AXP_s011
                                 + 1.28658 w_k1_DIS_s011
                                 + 1.38409 w_k1_FDX_s011
                                 + 0.798004 w_k1_IBM_s011
                                 + 1.06572 w_k1_INTC_s011
                                 + 1.02532 w_k1_KO_s011
                                 + 0.772862 w_k1_MAT_s011
                                 + 1.19154 w_k1_MSFT_s011
                                 + 1.09616 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.969566 w_k1_ADM_s011
                                 + 0.811421 w_k1_AXP_s011
                                 + 0.708143 w_k1_DIS_s011
                                 + 0.969918 w_k1_FDX_s011
                                 + 0.581752 w_k1_IBM_s011
                                 + 1.01906 w_k1_INTC_s011
                                 + 0.65782 w_k1_KO_s011
                                 + 0.517938 w_k1_MAT_s011
                                 + 0.918948 w_k1_MSFT_s011
                                 + 0.802136 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.857803 w_k1_ADM_s011
                                 + 0.902844 w_k1_AXP_s011
                                 + 0.805931 w_k1_DIS_s011
                                 + 0.839518 w_k1_FDX_s011
                                 + 0.933424 w_k1_IBM_s011
                                 + 0.961785 w_k1_INTC_s011
                                 + 0.846049 w_k1_KO_s011
                                 + 0.719178 w_k1_MAT_s011
                                 + 0.844741 w_k1_MSFT_s011
                                 + 0.91633 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.994224 w_k1_ADM_s011
                                 + 1.141 w_k1_AXP_s011 + 1.03933 w_k1_DIS_s011
                                 + 1.09264 w_k1_FDX_s011 + 1.0027 w_k1_IBM_s011
                                 + 1.14631 w_k1_INTC_s011
                                 + 1.04556 w_k1_KO_s011
                                 + 0.843802 w_k1_MAT_s011
                                 + 1.14173 w_k1_MSFT_s011
                                 + 1.17039 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.17963 w_k1_ADM_s011
                                 + 1.3158 w_k1_AXP_s011 + 1.59505 w_k1_DIS_s011
                                 + 1.31047 w_k1_FDX_s011
                                 + 1.44785 w_k1_IBM_s011
                                 + 1.50279 w_k1_INTC_s011
                                 + 1.59448 w_k1_KO_s011 + 1.83789 w_k1_MAT_s011
                                 + 1.61758 w_k1_MSFT_s011
                                 + 1.28461 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.853262 w_k1_ADM_s012
                                 + 0.829982 w_k1_AXP_s012
                                 + 0.675812 w_k1_DIS_s012
                                 + 0.725328 w_k1_FDX_s012
                                 + 1.11277 w_k1_IBM_s012
                                 + 0.735885 w_k1_INTC_s012
                                 + 0.834911 w_k1_KO_s012 + 1.016 w_k1_MAT_s012
                                 + 0.81204 w_k1_MSFT_s012 + 0.7685 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.14656 w_k1_ADM_s012
                                 + 0.973767 w_k1_AXP_s012
                                 + 0.957109 w_k1_DIS_s012
                                 + 0.94114 w_k1_FDX_s012
                                 + 1.35565 w_k1_IBM_s012
                                 + 0.798897 w_k1_INTC_s012
                                 + 1.08822 w_k1_KO_s012 + 1.09474 w_k1_MAT_s012
                                 + 0.883302 w_k1_MSFT_s012
                                 + 0.858012 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.998507 w_k1_ADM_s012
                                 + 1.46833 w_k1_AXP_s012
                                 + 1.51259 w_k1_DIS_s012
                                 + 1.25269 w_k1_FDX_s012
                                 + 1.36333 w_k1_IBM_s012
                                 + 1.15396 w_k1_INTC_s012
                                 + 1.31294 w_k1_KO_s012 + 1.60588 w_k1_MAT_s012
                                 + 1.38439 w_k1_MSFT_s012
                                 + 0.945142 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.811646 w_k1_ADM_s012
                                 + 0.816222 w_k1_AXP_s012
                                 + 0.744332 w_k1_DIS_s012
                                 + 0.806616 w_k1_FDX_s012
                                 + 0.879455 w_k1_IBM_s012
                                 + 0.878506 w_k1_INTC_s012
                                 + 0.884865 w_k1_KO_s012
                                 + 0.846588 w_k1_MAT_s012
                                 + 1.00181 w_k1_MSFT_s012
                                 + 0.871291 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.18558 w_k1_ADM_s012
                                 + 0.964743 w_k1_AXP_s012
                                 + 1.23109 w_k1_DIS_s012
                                 + 1.12836 w_k1_FDX_s012
                                 + 0.835598 w_k1_IBM_s012
                                 + 0.978815 w_k1_INTC_s012
                                 + 1.17513 w_k1_KO_s012 + 1.17872 w_k1_MAT_s012
                                 + 1.30701 w_k1_MSFT_s012
                                 + 1.01196 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.18514 w_k1_ADM_s012
                                 + 1.20125 w_k1_AXP_s012
                                 + 1.27903 w_k1_DIS_s012
                                 + 1.15401 w_k1_FDX_s012
                                 + 1.15246 w_k1_IBM_s012
                                 + 1.14907 w_k1_INTC_s012
                                 + 1.14147 w_k1_KO_s012 + 1.25239 w_k1_MAT_s012
                                 + 1.11459 w_k1_MSFT_s012 + 1.0949 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.07209 w_k1_ADM_s012
                                 + 0.996729 w_k1_AXP_s012
                                 + 0.885079 w_k1_DIS_s012
                                 + 1.08284 w_k1_FDX_s012
                                 + 0.788768 w_k1_IBM_s012
                                 + 0.891278 w_k1_INTC_s012
                                 + 0.846676 w_k1_KO_s012
                                 + 0.838244 w_k1_MAT_s012
                                 + 1.03136 w_k1_MSFT_s012
                                 + 0.927363 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.88826 w_k1_ADM_s012
                                 + 0.883561 w_k1_AXP_s012
                                 + 1.19622 w_k1_DIS_s012
                                 + 0.845743 w_k1_FDX_s012
                                 + 1.22602 w_k1_IBM_s012
                                 + 0.881774 w_k1_INTC_s012
                                 + 1.36297 w_k1_KO_s012 + 1.9763 w_k1_MAT_s012
                                 + 1.10185 w_k1_MSFT_s012
                                 + 1.04271 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.85035 w_k1_ADM_s013
                                 + 0.786952 w_k1_AXP_s013
                                 + 0.643268 w_k1_DIS_s013
                                 + 0.812095 w_k1_FDX_s013
                                 + 0.784453 w_k1_IBM_s013
                                 + 0.827579 w_k1_INTC_s013
                                 + 0.71058 w_k1_KO_s013
                                 + 0.569007 w_k1_MAT_s013
                                 + 0.695955 w_k1_MSFT_s013
                                 + 0.821515 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.02699 w_k1_ADM_s013
                                 + 1.02708 w_k1_AXP_s013
                                 + 0.840651 w_k1_DIS_s013
                                 + 1.00807 w_k1_FDX_s013
                                 + 0.702047 w_k1_IBM_s013
                                 + 0.886766 w_k1_INTC_s013
                                 + 0.799785 w_k1_KO_s013
                                 + 0.588113 w_k1_MAT_s013
                                 + 0.812855 w_k1_MSFT_s013
                                 + 0.827247 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.09273 w_k1_ADM_s013
                                 + 1.02905 w_k1_AXP_s013
                                 + 0.805979 w_k1_DIS_s013
                                 + 0.915863 w_k1_FDX_s013
                                 + 1.24373 w_k1_IBM_s013
                                 + 0.842399 w_k1_INTC_s013
                                 + 1.02229 w_k1_KO_s013 + 1.45108 w_k1_MAT_s013
                                 + 0.792885 w_k1_MSFT_s013
                                 + 1.10069 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.10761 w_k1_ADM_s013
                                 + 1.38251 w_k1_AXP_s013
                                 + 1.26285 w_k1_DIS_s013
                                 + 1.15718 w_k1_FDX_s013
                                 + 1.03465 w_k1_IBM_s013
                                 + 1.10417 w_k1_INTC_s013
                                 + 1.14796 w_k1_KO_s013 + 0.93939 w_k1_MAT_s013
                                 + 1.09384 w_k1_MSFT_s013
                                 + 0.938424 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06938 w_k1_ADM_s013
                                 + 1.49727 w_k1_AXP_s013
                                 + 1.63384 w_k1_DIS_s013
                                 + 1.25768 w_k1_FDX_s013
                                 + 1.81903 w_k1_IBM_s013
                                 + 1.07517 w_k1_INTC_s013
                                 + 1.51718 w_k1_KO_s013 + 1.63054 w_k1_MAT_s013
                                 + 1.06013 w_k1_MSFT_s013
                                 + 1.21164 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.08914 w_k1_ADM_s013
                                 + 1.39788 w_k1_AXP_s013
                                 + 1.67552 w_k1_DIS_s013
                                 + 1.44598 w_k1_FDX_s013
                                 + 0.900777 w_k1_IBM_s013
                                 + 1.27079 w_k1_INTC_s013
                                 + 1.14752 w_k1_KO_s013 + 1.39448 w_k1_MAT_s013
                                 + 1.46489 w_k1_MSFT_s013
                                 + 1.07591 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.778108 w_k1_ADM_s013
                                 + 0.89477 w_k1_AXP_s013 + 1.1244 w_k1_DIS_s013
                                 + 0.897645 w_k1_FDX_s013
                                 + 1.29074 w_k1_IBM_s013
                                 + 1.02664 w_k1_INTC_s013
                                 + 1.04269 w_k1_KO_s013 + 1.12324 w_k1_MAT_s013
                                 + 0.987709 w_k1_MSFT_s013
                                 + 0.987175 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.729564 w_k1_ADM_s013
                                 + 0.820329 w_k1_AXP_s013
                                 + 0.811346 w_k1_DIS_s013
                                 + 0.79657 w_k1_FDX_s013
                                 + 1.32882 w_k1_IBM_s013
                                 + 0.732928 w_k1_INTC_s013
                                 + 0.989685 w_k1_KO_s013
                                 + 0.902577 w_k1_MAT_s013
                                 + 0.753509 w_k1_MSFT_s013
                                 + 0.911494 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.19068 w_k1_ADM_s014
                                 + 1.16089 w_k1_AXP_s014
                                 + 1.34023 w_k1_DIS_s014
                                 + 1.18264 w_k1_FDX_s014
                                 + 0.814925 w_k1_IBM_s014
                                 + 1.1219 w_k1_INTC_s014 + 1.10322 w_k1_KO_s014
                                 + 0.952291 w_k1_MAT_s014
                                 + 1.23195 w_k1_MSFT_s014
                                 + 1.13251 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.03091 w_k1_ADM_s014
                                 + 1.57711 w_k1_AXP_s014
                                 + 1.07081 w_k1_DIS_s014
                                 + 1.11617 w_k1_FDX_s014
                                 + 1.16277 w_k1_IBM_s014
                                 + 1.36637 w_k1_INTC_s014
                                 + 1.16634 w_k1_KO_s014 + 1.81812 w_k1_MAT_s014
                                 + 1.13336 w_k1_MSFT_s014
                                 + 1.28389 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.979399 w_k1_ADM_s014
                                 + 0.886135 w_k1_AXP_s014
                                 + 0.841785 w_k1_DIS_s014
                                 + 0.976697 w_k1_FDX_s014
                                 + 0.739217 w_k1_IBM_s014
                                 + 1.06997 w_k1_INTC_s014
                                 + 0.935211 w_k1_KO_s014
                                 + 0.443001 w_k1_MAT_s014
                                 + 0.898738 w_k1_MSFT_s014
                                 + 1.10129 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.932068 w_k1_ADM_s014
                                 + 1.16092 w_k1_AXP_s014
                                 + 0.882768 w_k1_DIS_s014
                                 + 0.930793 w_k1_FDX_s014
                                 + 1.05888 w_k1_IBM_s014
                                 + 1.18257 w_k1_INTC_s014 + 1.049 w_k1_KO_s014
                                 + 1.10285 w_k1_MAT_s014
                                 + 0.941978 w_k1_MSFT_s014
                                 + 0.966343 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.844363 w_k1_ADM_s014
                                 + 1.02315 w_k1_AXP_s014
                                 + 1.15007 w_k1_DIS_s014
                                 + 0.885257 w_k1_FDX_s014
                                 + 1.25153 w_k1_IBM_s014
                                 + 1.01208 w_k1_INTC_s014
                                 + 1.29918 w_k1_KO_s014 + 1.36638 w_k1_MAT_s014
                                 + 0.896706 w_k1_MSFT_s014
                                 + 1.1807 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.733582 w_k1_ADM_s014
                                 + 0.661997 w_k1_AXP_s014
                                 + 0.72277 w_k1_DIS_s014
                                 + 0.714678 w_k1_FDX_s014
                                 + 0.836681 w_k1_IBM_s014
                                 + 0.891784 w_k1_INTC_s014
                                 + 0.806094 w_k1_KO_s014
                                 + 0.736765 w_k1_MAT_s014
                                 + 0.828429 w_k1_MSFT_s014
                                 + 0.847786 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.43716 w_k1_ADM_s014
                                 + 1.49705 w_k1_AXP_s014
                                 + 1.26526 w_k1_DIS_s014
                                 + 1.11146 w_k1_FDX_s014
                                 + 1.62082 w_k1_IBM_s014
                                 + 1.28339 w_k1_INTC_s014
                                 + 1.43327 w_k1_KO_s014 + 1.21879 w_k1_MAT_s014
                                 + 1.04064 w_k1_MSFT_s014 + 1.2734 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.34945 w_k1_ADM_s014
                                 + 1.47031 w_k1_AXP_s014 + 1.6743 w_k1_DIS_s014
                                 + 1.42091 w_k1_FDX_s014
                                 + 1.18949 w_k1_IBM_s014
                                 + 1.20015 w_k1_INTC_s014 + 1.2594 w_k1_KO_s014
                                 + 1.06089 w_k1_MAT_s014
                                 + 1.33024 w_k1_MSFT_s014
                                 + 1.51334 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.19936 w_k1_ADM_s015
                                 + 0.766545 w_k1_AXP_s015
                                 + 0.717643 w_k1_DIS_s015
                                 + 0.929773 w_k1_FDX_s015
                                 + 0.673406 w_k1_IBM_s015
                                 + 0.860738 w_k1_INTC_s015
                                 + 0.758163 w_k1_KO_s015
                                 + 0.535275 w_k1_MAT_s015
                                 + 0.898399 w_k1_MSFT_s015
                                 + 0.897151 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.910657 w_k1_ADM_s015
                                 + 0.726958 w_k1_AXP_s015
                                 + 0.946116 w_k1_DIS_s015
                                 + 1.02415 w_k1_FDX_s015
                                 + 0.702212 w_k1_IBM_s015
                                 + 1.07368 w_k1_INTC_s015
                                 + 0.822807 w_k1_KO_s015
                                 + 0.454169 w_k1_MAT_s015
                                 + 0.991018 w_k1_MSFT_s015
                                 + 1.04596 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.07311 w_k1_ADM_s015
                                 + 1.25601 w_k1_AXP_s015
                                 + 0.973743 w_k1_DIS_s015
                                 + 1.01319 w_k1_FDX_s015
                                 + 0.825031 w_k1_IBM_s015
                                 + 0.707158 w_k1_INTC_s015
                                 + 0.78749 w_k1_KO_s015 + 1.07153 w_k1_MAT_s015
                                 + 0.844899 w_k1_MSFT_s015
                                 + 1.04044 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.825589 w_k1_ADM_s015
                                 + 0.82774 w_k1_AXP_s015
                                 + 0.637963 w_k1_DIS_s015
                                 + 0.753823 w_k1_FDX_s015
                                 + 0.901407 w_k1_IBM_s015
                                 + 0.857018 w_k1_INTC_s015
                                 + 0.867509 w_k1_KO_s015
                                 + 0.78053 w_k1_MAT_s015
                                 + 0.857942 w_k1_MSFT_s015
                                 + 0.931849 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.01222 w_k1_ADM_s015
                                 + 1.1524 w_k1_AXP_s015 + 1.06488 w_k1_DIS_s015
                                 + 1.07785 w_k1_FDX_s015 + 1.1639 w_k1_IBM_s015
                                 + 0.912578 w_k1_INTC_s015
                                 + 1.03233 w_k1_KO_s015 + 1.67059 w_k1_MAT_s015
                                 + 1.00414 w_k1_MSFT_s015
                                 + 1.14218 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.06686 w_k1_ADM_s015
                                 + 1.17396 w_k1_AXP_s015
                                 + 0.900305 w_k1_DIS_s015
                                 + 0.868453 w_k1_FDX_s015
                                 + 1.16861 w_k1_IBM_s015
                                 + 0.941212 w_k1_INTC_s015
                                 + 0.932491 w_k1_KO_s015
                                 + 1.51087 w_k1_MAT_s015
                                 + 1.0033 w_k1_MSFT_s015
                                 + 0.996267 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.21849 w_k1_ADM_s015
                                 + 1.2633 w_k1_AXP_s015 + 1.47884 w_k1_DIS_s015
                                 + 1.27167 w_k1_FDX_s015
                                 + 1.08373 w_k1_IBM_s015
                                 + 1.18461 w_k1_INTC_s015 + 1.0742 w_k1_KO_s015
                                 + 0.945611 w_k1_MAT_s015
                                 + 1.31597 w_k1_MSFT_s015
                                 + 1.13799 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.1475 w_k1_ADM_s015
                                 + 1.64306 w_k1_AXP_s015
                                 + 1.20865 w_k1_DIS_s015
                                 + 1.30552 w_k1_FDX_s015
                                 + 1.05185 w_k1_IBM_s015
                                 + 1.07589 w_k1_INTC_s015
                                 + 0.923886 w_k1_KO_s015 + 1.5951 w_k1_MAT_s015
                                 + 1.06821 w_k1_MSFT_s015
                                 + 1.12218 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.17192 w_k1_ADM_s016
                                 + 1.49571 w_k1_AXP_s016
                                 + 1.53291 w_k1_DIS_s016
                                 + 1.44365 w_k1_FDX_s016
                                 + 1.15339 w_k1_IBM_s016
                                 + 1.38834 w_k1_INTC_s016
                                 + 1.47823 w_k1_KO_s016 + 2.29849 w_k1_MAT_s016
                                 + 1.3524 w_k1_MSFT_s016 + 1.26973 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.939297 w_k1_ADM_s016
                                 + 0.76934 w_k1_AXP_s016
                                 + 1.13241 w_k1_DIS_s016
                                 + 0.991627 w_k1_FDX_s016
                                 + 1.07752 w_k1_IBM_s016
                                 + 0.992933 w_k1_INTC_s016
                                 + 1.26011 w_k1_KO_s016 + 1.11481 w_k1_MAT_s016
                                 + 1.22116 w_k1_MSFT_s016
                                 + 0.994334 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.688767 w_k1_ADM_s016
                                 + 1.37387 w_k1_AXP_s016
                                 + 1.15326 w_k1_DIS_s016
                                 + 1.03074 w_k1_FDX_s016
                                 + 1.62303 w_k1_IBM_s016
                                 + 1.0138 w_k1_INTC_s016 + 1.41111 w_k1_KO_s016
                                 + 3.49558 w_k1_MAT_s016
                                 + 0.977324 w_k1_MSFT_s016
                                 + 1.10056 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.853684 w_k1_ADM_s016
                                 + 0.988476 w_k1_AXP_s016
                                 + 1.26051 w_k1_DIS_s016
                                 + 1.14889 w_k1_FDX_s016
                                 + 1.19749 w_k1_IBM_s016
                                 + 1.24713 w_k1_INTC_s016
                                 + 1.39337 w_k1_KO_s016 + 1.54191 w_k1_MAT_s016
                                 + 1.21485 w_k1_MSFT_s016
                                 + 1.07134 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.757247 w_k1_ADM_s016
                                 + 0.784177 w_k1_AXP_s016
                                 + 0.858693 w_k1_DIS_s016
                                 + 0.909236 w_k1_FDX_s016
                                 + 0.809418 w_k1_IBM_s016
                                 + 1.11243 w_k1_INTC_s016
                                 + 0.972008 w_k1_KO_s016
                                 + 0.973833 w_k1_MAT_s016
                                 + 1.06799 w_k1_MSFT_s016
                                 + 0.787898 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.832252 w_k1_ADM_s016
                                 + 0.955363 w_k1_AXP_s016
                                 + 0.913837 w_k1_DIS_s016
                                 + 1.02704 w_k1_FDX_s016
                                 + 0.886265 w_k1_IBM_s016
                                 + 1.11227 w_k1_INTC_s016
                                 + 1.05429 w_k1_KO_s016 + 1.29194 w_k1_MAT_s016
                                 + 1.10062 w_k1_MSFT_s016
                                 + 1.02118 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.724723 w_k1_ADM_s016
                                 + 0.579291 w_k1_AXP_s016
                                 + 0.608717 w_k1_DIS_s016
                                 + 0.787305 w_k1_FDX_s016
                                 + 0.725294 w_k1_IBM_s016
                                 + 0.69502 w_k1_INTC_s016
                                 + 0.752373 w_k1_KO_s016
                                 + 0.609418 w_k1_MAT_s016
                                 + 0.746847 w_k1_MSFT_s016
                                 + 0.738391 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.750203 w_k1_ADM_s016
                                 + 0.946713 w_k1_AXP_s016
                                 + 0.839354 w_k1_DIS_s016
                                 + 0.862869 w_k1_FDX_s016
                                 + 1.30053 w_k1_IBM_s016
                                 + 0.892669 w_k1_INTC_s016
                                 + 1.09843 w_k1_KO_s016 + 1.66552 w_k1_MAT_s016
                                 + 0.970396 w_k1_MSFT_s016
                                 + 0.949225 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.11304 w_k1_ADM_s017
                                 + 1.31554 w_k1_AXP_s017
                                 + 1.70606 w_k1_DIS_s017
                                 + 1.24367 w_k1_FDX_s017
                                 + 1.02802 w_k1_IBM_s017
                                 + 1.50491 w_k1_INTC_s017
                                 + 1.26378 w_k1_KO_s017
                                 + 0.683554 w_k1_MAT_s017
                                 + 1.56647 w_k1_MSFT_s017
                                 + 1.24336 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.847556 w_k1_ADM_s017
                                 + 0.744965 w_k1_AXP_s017
                                 + 0.898942 w_k1_DIS_s017
                                 + 0.837118 w_k1_FDX_s017
                                 + 1.32185 w_k1_IBM_s017
                                 + 1.34104 w_k1_INTC_s017
                                 + 1.34207 w_k1_KO_s017 + 1.10572 w_k1_MAT_s017
                                 + 1.16431 w_k1_MSFT_s017
                                 + 1.05894 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.818882 w_k1_ADM_s017
                                 + 0.850953 w_k1_AXP_s017
                                 + 0.910361 w_k1_DIS_s017
                                 + 0.937066 w_k1_FDX_s017
                                 + 0.887086 w_k1_IBM_s017
                                 + 1.1645 w_k1_INTC_s017 + 1.0098 w_k1_KO_s017
                                 + 0.787282 w_k1_MAT_s017
                                 + 1.08892 w_k1_MSFT_s017
                                 + 1.16984 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.53207 w_k1_ADM_s017
                                 + 1.48005 w_k1_AXP_s017
                                 + 1.19231 w_k1_DIS_s017
                                 + 1.30153 w_k1_FDX_s017
                                 + 0.814484 w_k1_IBM_s017
                                 + 1.40433 w_k1_INTC_s017
                                 + 1.07021 w_k1_KO_s017
                                 + 0.999423 w_k1_MAT_s017
                                 + 1.45712 w_k1_MSFT_s017
                                 + 1.23821 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.732363 w_k1_ADM_s017
                                 + 0.726918 w_k1_AXP_s017
                                 + 0.626796 w_k1_DIS_s017
                                 + 0.812673 w_k1_FDX_s017
                                 + 0.670744 w_k1_IBM_s017
                                 + 1.01724 w_k1_INTC_s017
                                 + 0.859147 w_k1_KO_s017
                                 + 0.753973 w_k1_MAT_s017
                                 + 0.945197 w_k1_MSFT_s017
                                 + 0.99301 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.08139 w_k1_ADM_s017
                                 + 1.15645 w_k1_AXP_s017
                                 + 1.32704 w_k1_DIS_s017 + 1.2075 w_k1_FDX_s017
                                 + 1.21015 w_k1_IBM_s017
                                 + 1.39928 w_k1_INTC_s017
                                 + 1.32636 w_k1_KO_s017 + 1.14826 w_k1_MAT_s017
                                 + 1.23961 w_k1_MSFT_s017 + 1.1658 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.04246 w_k1_ADM_s017
                                 + 1.31723 w_k1_AXP_s017
                                 + 0.959173 w_k1_DIS_s017
                                 + 0.955643 w_k1_FDX_s017
                                 + 1.51453 w_k1_IBM_s017
                                 + 1.06862 w_k1_INTC_s017
                                 + 1.33619 w_k1_KO_s017 + 1.50215 w_k1_MAT_s017
                                 + 0.971144 w_k1_MSFT_s017
                                 + 1.15714 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.12611 w_k1_ADM_s017
                                 + 1.96713 w_k1_AXP_s017 + 1.8398 w_k1_DIS_s017
                                 + 1.49498 w_k1_FDX_s017 + 1.5975 w_k1_IBM_s017
                                 + 1.37614 w_k1_INTC_s017 + 1.5491 w_k1_KO_s017
                                 + 1.68511 w_k1_MAT_s017
                                 + 1.42988 w_k1_MSFT_s017
                                 + 1.46145 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.08296 w_k1_ADM_s100
                                 + 1.16911 w_k1_AXP_s100
                                 + 1.25907 w_k1_DIS_s100 + 1.2763 w_k1_FDX_s100
                                 + 0.853057 w_k1_IBM_s100
                                 + 1.30993 w_k1_INTC_s100
                                 + 1.10624 w_k1_KO_s100 + 1.06156 w_k1_MAT_s100
                                 + 1.32422 w_k1_MSFT_s100
                                 + 1.08698 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.34692 w_k1_ADM_s100
                                 + 1.37514 w_k1_AXP_s100
                                 + 1.30964 w_k1_DIS_s100 + 1.3072 w_k1_FDX_s100
                                 + 1.14739 w_k1_IBM_s100
                                 + 1.05763 w_k1_INTC_s100
                                 + 1.18278 w_k1_KO_s100 + 1.50665 w_k1_MAT_s100
                                 + 1.15042 w_k1_MSFT_s100 + 1.2024 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.634456 w_k1_ADM_s100
                                 + 0.667053 w_k1_AXP_s100
                                 + 0.608878 w_k1_DIS_s100
                                 + 0.656886 w_k1_FDX_s100
                                 + 1.04297 w_k1_IBM_s100
                                 + 0.984346 w_k1_INTC_s100
                                 + 0.95551 w_k1_KO_s100 + 0.83139 w_k1_MAT_s100
                                 + 0.879846 w_k1_MSFT_s100
                                 + 0.824338 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.09809 w_k1_ADM_s100
                                 + 0.926288 w_k1_AXP_s100
                                 + 1.05381 w_k1_DIS_s100
                                 + 1.10485 w_k1_FDX_s100
                                 + 1.01068 w_k1_IBM_s100
                                 + 1.20576 w_k1_INTC_s100
                                 + 1.01394 w_k1_KO_s100 + 1.11536 w_k1_MAT_s100
                                 + 1.10165 w_k1_MSFT_s100
                                 + 1.14481 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.991801 w_k1_ADM_s100
                                 + 0.771235 w_k1_AXP_s100
                                 + 0.75381 w_k1_DIS_s100
                                 + 0.955043 w_k1_FDX_s100
                                 + 0.698945 w_k1_IBM_s100
                                 + 0.813545 w_k1_INTC_s100
                                 + 0.716524 w_k1_KO_s100
                                 + 0.821942 w_k1_MAT_s100
                                 + 0.954539 w_k1_MSFT_s100
                                 + 0.861722 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.771677 w_k1_ADM_s100
                                 + 0.614099 w_k1_AXP_s100
                                 + 0.746537 w_k1_DIS_s100
                                 + 0.830183 w_k1_FDX_s100
                                 + 0.837445 w_k1_IBM_s100
                                 + 0.963868 w_k1_INTC_s100
                                 + 0.867771 w_k1_KO_s100
                                 + 0.614038 w_k1_MAT_s100
                                 + 0.881777 w_k1_MSFT_s100
                                 + 0.827686 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.22419 w_k1_ADM_s100
                                 + 1.45373 w_k1_AXP_s100
                                 + 1.66722 w_k1_DIS_s100
                                 + 1.25888 w_k1_FDX_s100 + 1.5715 w_k1_IBM_s100
                                 + 1.31002 w_k1_INTC_s100
                                 + 1.56689 w_k1_KO_s100 + 2.09269 w_k1_MAT_s100
                                 + 1.28622 w_k1_MSFT_s100
                                 + 1.19107 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.813649 w_k1_ADM_s100
                                 + 0.918238 w_k1_AXP_s100
                                 + 0.794088 w_k1_DIS_s100
                                 + 0.890895 w_k1_FDX_s100
                                 + 1.29814 w_k1_IBM_s100
                                 + 0.917478 w_k1_INTC_s100
                                 + 1.04204 w_k1_KO_s100 + 1.21981 w_k1_MAT_s100
                                 + 0.949901 w_k1_MSFT_s100
                                 + 0.90557 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.09994 w_k1_ADM_s101
                                 + 1.09631 w_k1_AXP_s101
                                 + 0.924053 w_k1_DIS_s101
                                 + 1.02304 w_k1_FDX_s101
                                 + 1.38864 w_k1_IBM_s101
                                 + 1.06357 w_k1_INTC_s101
                                 + 0.966907 w_k1_KO_s101
                                 + 1.21022 w_k1_MAT_s101
                                 + 1.03623 w_k1_MSFT_s101
                                 + 0.90942 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.70955 w_k1_ADM_s101
                                 + 0.916131 w_k1_AXP_s101
                                 + 0.858957 w_k1_DIS_s101
                                 + 0.886226 w_k1_FDX_s101
                                 + 1.02617 w_k1_IBM_s101
                                 + 1.18109 w_k1_INTC_s101
                                 + 0.908314 w_k1_KO_s101
                                 + 0.946235 w_k1_MAT_s101
                                 + 1.02679 w_k1_MSFT_s101
                                 + 0.883615 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.14921 w_k1_ADM_s101
                                 + 1.34831 w_k1_AXP_s101
                                 + 0.968149 w_k1_DIS_s101
                                 + 1.1132 w_k1_FDX_s101 + 1.10584 w_k1_IBM_s101
                                 + 1.21077 w_k1_INTC_s101
                                 + 0.957201 w_k1_KO_s101
                                 + 1.08575 w_k1_MAT_s101
                                 + 1.04699 w_k1_MSFT_s101
                                 + 1.03291 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.10507 w_k1_ADM_s101
                                 + 1.1777 w_k1_AXP_s101
                                 + 0.995581 w_k1_DIS_s101
                                 + 1.14594 w_k1_FDX_s101
                                 + 0.737212 w_k1_IBM_s101
                                 + 1.11995 w_k1_INTC_s101
                                 + 0.758211 w_k1_KO_s101
                                 + 0.751123 w_k1_MAT_s101
                                 + 1.05587 w_k1_MSFT_s101
                                 + 0.94226 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.04208 w_k1_ADM_s101
                                 + 1.01843 w_k1_AXP_s101
                                 + 1.38712 w_k1_DIS_s101
                                 + 1.12995 w_k1_FDX_s101
                                 + 1.71703 w_k1_IBM_s101
                                 + 1.30059 w_k1_INTC_s101
                                 + 1.35282 w_k1_KO_s101 + 1.32723 w_k1_MAT_s101
                                 + 1.21726 w_k1_MSFT_s101
                                 + 1.32681 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.75555 w_k1_ADM_s101
                                 + 0.592114 w_k1_AXP_s101
                                 + 0.607623 w_k1_DIS_s101
                                 + 0.785818 w_k1_FDX_s101
                                 + 0.667444 w_k1_IBM_s101
                                 + 0.814428 w_k1_INTC_s101
                                 + 0.581752 w_k1_KO_s101
                                 + 0.468146 w_k1_MAT_s101
                                 + 0.814004 w_k1_MSFT_s101
                                 + 0.75459 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.956922 w_k1_ADM_s101
                                 + 0.709237 w_k1_AXP_s101
                                 + 0.916957 w_k1_DIS_s101
                                 + 1.02322 w_k1_FDX_s101
                                 + 0.720541 w_k1_IBM_s101
                                 + 1.08985 w_k1_INTC_s101
                                 + 0.776499 w_k1_KO_s101
                                 + 0.626523 w_k1_MAT_s101
                                 + 1.05322 w_k1_MSFT_s101
                                 + 1.03254 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.25921 w_k1_ADM_s101
                                 + 1.48899 w_k1_AXP_s101
                                 + 1.70106 w_k1_DIS_s101
                                 + 1.32207 w_k1_FDX_s101
                                 + 1.19456 w_k1_IBM_s101
                                 + 1.11456 w_k1_INTC_s101
                                 + 1.25271 w_k1_KO_s101 + 1.23941 w_k1_MAT_s101
                                 + 1.30427 w_k1_MSFT_s101
                                 + 1.31604 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.69204 w_k1_ADM_s102
                                 + 0.700714 w_k1_AXP_s102
                                 + 0.541214 w_k1_DIS_s102
                                 + 0.717615 w_k1_FDX_s102
                                 + 0.801168 w_k1_IBM_s102
                                 + 0.701171 w_k1_INTC_s102
                                 + 0.621599 w_k1_KO_s102
                                 + 0.713255 w_k1_MAT_s102
                                 + 0.642416 w_k1_MSFT_s102
                                 + 0.732484 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.848007 w_k1_ADM_s102
                                 + 0.75077 w_k1_AXP_s102
                                 + 0.894006 w_k1_DIS_s102
                                 + 0.862055 w_k1_FDX_s102
                                 + 1.19486 w_k1_IBM_s102
                                 + 1.24345 w_k1_INTC_s102
                                 + 0.989714 w_k1_KO_s102
                                 + 0.73099 w_k1_MAT_s102
                                 + 0.991623 w_k1_MSFT_s102
                                 + 0.90519 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.21556 w_k1_ADM_s102
                                 + 1.24684 w_k1_AXP_s102
                                 + 1.29281 w_k1_DIS_s102
                                 + 1.12696 w_k1_FDX_s102 + 1.0817 w_k1_IBM_s102
                                 + 1.13626 w_k1_INTC_s102
                                 + 1.34096 w_k1_KO_s102
                                 + 0.942209 w_k1_MAT_s102
                                 + 1.09814 w_k1_MSFT_s102
                                 + 1.04364 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.04226 w_k1_ADM_s102
                                 + 1.17617 w_k1_AXP_s102
                                 + 1.17208 w_k1_DIS_s102
                                 + 1.03764 w_k1_FDX_s102
                                 + 1.52087 w_k1_IBM_s102
                                 + 1.36645 w_k1_INTC_s102
                                 + 1.27262 w_k1_KO_s102 + 1.18054 w_k1_MAT_s102
                                 + 1.06365 w_k1_MSFT_s102
                                 + 1.10619 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.961921 w_k1_ADM_s102
                                 + 0.86825 w_k1_AXP_s102
                                 + 0.737534 w_k1_DIS_s102
                                 + 0.95045 w_k1_FDX_s102
                                 + 0.779828 w_k1_IBM_s102
                                 + 1.18779 w_k1_INTC_s102
                                 + 0.83834 w_k1_KO_s102
                                 + 0.617434 w_k1_MAT_s102
                                 + 0.953021 w_k1_MSFT_s102
                                 + 0.914205 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.22389 w_k1_ADM_s102
                                 + 1.48146 w_k1_AXP_s102
                                 + 1.66514 w_k1_DIS_s102 + 1.3817 w_k1_FDX_s102
                                 + 1.38409 w_k1_IBM_s102
                                 + 1.36483 w_k1_INTC_s102
                                 + 1.33019 w_k1_KO_s102 + 1.33585 w_k1_MAT_s102
                                 + 1.25907 w_k1_MSFT_s102
                                 + 1.19022 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.717133 w_k1_ADM_s102
                                 + 0.869177 w_k1_AXP_s102
                                 + 0.88404 w_k1_DIS_s102
                                 + 0.836452 w_k1_FDX_s102
                                 + 0.967895 w_k1_IBM_s102
                                 + 0.850289 w_k1_INTC_s102
                                 + 0.965855 w_k1_KO_s102
                                 + 0.80101 w_k1_MAT_s102
                                 + 0.878618 w_k1_MSFT_s102
                                 + 0.934561 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.980569 w_k1_ADM_s102
                                 + 0.944431 w_k1_AXP_s102
                                 + 1.23035 w_k1_DIS_s102
                                 + 1.11533 w_k1_FDX_s102
                                 + 0.926712 w_k1_IBM_s102
                                 + 1.262 w_k1_INTC_s102 + 0.997426 w_k1_KO_s102
                                 + 0.652958 w_k1_MAT_s102
                                 + 1.20111 w_k1_MSFT_s102
                                 + 1.02898 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.27129 w_k1_ADM_s103
                                 + 1.24315 w_k1_AXP_s103 + 1.2516 w_k1_DIS_s103
                                 + 1.14582 w_k1_FDX_s103
                                 + 1.08524 w_k1_IBM_s103
                                 + 1.11377 w_k1_INTC_s103
                                 + 0.956927 w_k1_KO_s103
                                 + 1.14061 w_k1_MAT_s103
                                 + 1.02552 w_k1_MSFT_s103
                                 + 0.968679 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.09525 w_k1_ADM_s103
                                 + 1.11724 w_k1_AXP_s103 + 1.4017 w_k1_DIS_s103
                                 + 1.10684 w_k1_FDX_s103 + 1.6118 w_k1_IBM_s103
                                 + 1.25832 w_k1_INTC_s103
                                 + 1.24616 w_k1_KO_s103 + 1.52592 w_k1_MAT_s103
                                 + 1.02204 w_k1_MSFT_s103 + 1.149 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.767938 w_k1_ADM_s103
                                 + 0.936807 w_k1_AXP_s103
                                 + 1.04826 w_k1_DIS_s103
                                 + 0.906221 w_k1_FDX_s103
                                 + 1.28145 w_k1_IBM_s103
                                 + 0.913672 w_k1_INTC_s103
                                 + 0.937132 w_k1_KO_s103
                                 + 0.981613 w_k1_MAT_s103
                                 + 0.869377 w_k1_MSFT_s103
                                 + 0.990536 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.787961 w_k1_ADM_s103
                                 + 0.806306 w_k1_AXP_s103
                                 + 0.723526 w_k1_DIS_s103
                                 + 0.85477 w_k1_FDX_s103
                                 + 0.688662 w_k1_IBM_s103
                                 + 0.914095 w_k1_INTC_s103
                                 + 0.658358 w_k1_KO_s103
                                 + 0.800521 w_k1_MAT_s103
                                 + 0.662877 w_k1_MSFT_s103
                                 + 0.860949 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.1859 w_k1_ADM_s103
                                 + 0.786556 w_k1_AXP_s103
                                 + 1.00507 w_k1_DIS_s103
                                 + 0.91093 w_k1_FDX_s103
                                 + 0.912837 w_k1_IBM_s103
                                 + 0.971929 w_k1_INTC_s103
                                 + 0.994044 w_k1_KO_s103
                                 + 0.903199 w_k1_MAT_s103
                                 + 0.882496 w_k1_MSFT_s103
                                 + 0.990551 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.29529 w_k1_ADM_s103
                                 + 1.34891 w_k1_AXP_s103 + 1.2806 w_k1_DIS_s103
                                 + 1.23551 w_k1_FDX_s103
                                 + 0.911024 w_k1_IBM_s103
                                 + 1.25822 w_k1_INTC_s103
                                 + 0.996398 w_k1_KO_s103
                                 + 1.07199 w_k1_MAT_s103
                                 + 1.27223 w_k1_MSFT_s103
                                 + 1.16113 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.14393 w_k1_ADM_s103
                                 + 1.07284 w_k1_AXP_s103 + 1.0315 w_k1_DIS_s103
                                 + 1.20425 w_k1_FDX_s103
                                 + 0.648877 w_k1_IBM_s103
                                 + 1.15501 w_k1_INTC_s103
                                 + 0.816262 w_k1_KO_s103
                                 + 0.633761 w_k1_MAT_s103
                                 + 0.963488 w_k1_MSFT_s103
                                 + 0.993385 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.63821 w_k1_ADM_s103
                                 + 1.53887 w_k1_AXP_s103
                                 + 1.97232 w_k1_DIS_s103
                                 + 1.53292 w_k1_FDX_s103
                                 + 1.19224 w_k1_IBM_s103
                                 + 1.14627 w_k1_INTC_s103
                                 + 1.29788 w_k1_KO_s103 + 1.32054 w_k1_MAT_s103
                                 + 1.35237 w_k1_MSFT_s103
                                 + 1.26466 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.803859 w_k1_ADM_s104
                                 + 0.689639 w_k1_AXP_s104
                                 + 0.754418 w_k1_DIS_s104
                                 + 0.893594 w_k1_FDX_s104
                                 + 0.793258 w_k1_IBM_s104
                                 + 0.759969 w_k1_INTC_s104
                                 + 0.878806 w_k1_KO_s104
                                 + 0.633678 w_k1_MAT_s104
                                 + 0.933152 w_k1_MSFT_s104
                                 + 0.844375 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.08657 w_k1_ADM_s104
                                 + 0.8593 w_k1_AXP_s104 + 1.10483 w_k1_DIS_s104
                                 + 1.09748 w_k1_FDX_s104
                                 + 1.10999 w_k1_IBM_s104
                                 + 0.932667 w_k1_INTC_s104
                                 + 1.28965 w_k1_KO_s104 + 1.09613 w_k1_MAT_s104
                                 + 1.29983 w_k1_MSFT_s104
                                 + 1.10367 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.700832 w_k1_ADM_s104
                                 + 0.956277 w_k1_AXP_s104
                                 + 0.863416 w_k1_DIS_s104
                                 + 0.89069 w_k1_FDX_s104
                                 + 1.18584 w_k1_IBM_s104
                                 + 0.71717 w_k1_INTC_s104
                                 + 0.996665 w_k1_KO_s104 + 1.3439 w_k1_MAT_s104
                                 + 0.89602 w_k1_MSFT_s104
                                 + 1.03435 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.895869 w_k1_ADM_s104
                                 + 1.06936 w_k1_AXP_s104
                                 + 1.08902 w_k1_DIS_s104
                                 + 1.16034 w_k1_FDX_s104
                                 + 0.864525 w_k1_IBM_s104
                                 + 0.930167 w_k1_INTC_s104
                                 + 1.00106 w_k1_KO_s104 + 1.09588 w_k1_MAT_s104
                                 + 1.1891 w_k1_MSFT_s104 + 1.12881 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.888833 w_k1_ADM_s104
                                 + 1.00865 w_k1_AXP_s104
                                 + 0.983211 w_k1_DIS_s104
                                 + 0.995872 w_k1_FDX_s104
                                 + 1.42047 w_k1_IBM_s104
                                 + 0.952562 w_k1_INTC_s104
                                 + 1.1072 w_k1_KO_s104 + 1.6899 w_k1_MAT_s104
                                 + 1.07459 w_k1_MSFT_s104
                                 + 1.02079 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.753594 w_k1_ADM_s104
                                 + 0.678451 w_k1_AXP_s104
                                 + 0.532825 w_k1_DIS_s104
                                 + 0.693482 w_k1_FDX_s104
                                 + 0.733172 w_k1_IBM_s104
                                 + 0.765763 w_k1_INTC_s104
                                 + 0.755378 w_k1_KO_s104
                                 + 0.805498 w_k1_MAT_s104
                                 + 0.86692 w_k1_MSFT_s104
                                 + 0.813915 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.912824 w_k1_ADM_s104
                                 + 1.02269 w_k1_AXP_s104
                                 + 0.872182 w_k1_DIS_s104
                                 + 0.935225 w_k1_FDX_s104
                                 + 0.912474 w_k1_IBM_s104
                                 + 0.810633 w_k1_INTC_s104
                                 + 0.947659 w_k1_KO_s104
                                 + 1.10108 w_k1_MAT_s104
                                 + 1.01158 w_k1_MSFT_s104
                                 + 0.959433 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.21826 w_k1_ADM_s104
                                 + 1.40881 w_k1_AXP_s104
                                 + 1.55653 w_k1_DIS_s104
                                 + 1.34951 w_k1_FDX_s104
                                 + 1.41014 w_k1_IBM_s104
                                 + 1.27801 w_k1_INTC_s104
                                 + 1.27823 w_k1_KO_s104 + 1.14536 w_k1_MAT_s104
                                 + 1.35652 w_k1_MSFT_s104
                                 + 1.26863 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.87061 w_k1_ADM_s105
                                 + 1.35509 w_k1_AXP_s105 + 1.568 w_k1_DIS_s105
                                 + 1.4839 w_k1_FDX_s105
                                 + 0.951832 w_k1_IBM_s105
                                 + 1.03935 w_k1_INTC_s105
                                 + 1.35957 w_k1_KO_s105 + 1.30693 w_k1_MAT_s105
                                 + 1.36717 w_k1_MSFT_s105
                                 + 1.19046 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.849115 w_k1_ADM_s105
                                 + 0.616809 w_k1_AXP_s105
                                 + 0.532328 w_k1_DIS_s105
                                 + 0.693372 w_k1_FDX_s105
                                 + 0.749078 w_k1_IBM_s105
                                 + 0.705109 w_k1_INTC_s105
                                 + 0.723594 w_k1_KO_s105
                                 + 0.675299 w_k1_MAT_s105
                                 + 0.704498 w_k1_MSFT_s105
                                 + 0.82117 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.06661 w_k1_ADM_s105
                                 + 0.812372 w_k1_AXP_s105
                                 + 0.83388 w_k1_DIS_s105
                                 + 0.900175 w_k1_FDX_s105
                                 + 0.815475 w_k1_IBM_s105
                                 + 0.935595 w_k1_INTC_s105
                                 + 0.835074 w_k1_KO_s105
                                 + 0.724624 w_k1_MAT_s105
                                 + 0.830386 w_k1_MSFT_s105
                                 + 0.936667 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.06495 w_k1_ADM_s105
                                 + 0.700067 w_k1_AXP_s105
                                 + 0.532832 w_k1_DIS_s105
                                 + 0.651126 w_k1_FDX_s105
                                 + 1.30909 w_k1_IBM_s105
                                 + 0.77269 w_k1_INTC_s105
                                 + 0.913395 w_k1_KO_s105
                                 + 1.15213 w_k1_MAT_s105
                                 + 0.741221 w_k1_MSFT_s105
                                 + 0.843339 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.22424 w_k1_ADM_s105
                                 + 1.13012 w_k1_AXP_s105
                                 + 0.836565 w_k1_DIS_s105
                                 + 1.03883 w_k1_FDX_s105
                                 + 0.614635 w_k1_IBM_s105
                                 + 1.19878 w_k1_INTC_s105
                                 + 0.855559 w_k1_KO_s105
                                 + 0.771502 w_k1_MAT_s105
                                 + 1.03042 w_k1_MSFT_s105
                                 + 0.902757 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.06133 w_k1_ADM_s105
                                 + 0.977642 w_k1_AXP_s105
                                 + 0.849907 w_k1_DIS_s105
                                 + 0.872665 w_k1_FDX_s105
                                 + 1.02027 w_k1_IBM_s105
                                 + 0.946593 w_k1_INTC_s105
                                 + 1.07579 w_k1_KO_s105 + 1.18519 w_k1_MAT_s105
                                 + 0.958912 w_k1_MSFT_s105
                                 + 1.09715 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.14923 w_k1_ADM_s105
                                 + 1.04609 w_k1_AXP_s105
                                 + 1.15897 w_k1_DIS_s105
                                 + 0.989782 w_k1_FDX_s105
                                 + 1.17301 w_k1_IBM_s105
                                 + 1.28268 w_k1_INTC_s105
                                 + 1.21611 w_k1_KO_s105 + 1.2142 w_k1_MAT_s105
                                 + 1.074 w_k1_MSFT_s105 + 0.925672 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.26411 w_k1_ADM_s105
                                 + 1.35654 w_k1_AXP_s105
                                 + 1.12357 w_k1_DIS_s105
                                 + 1.04724 w_k1_FDX_s105
                                 + 1.47325 w_k1_IBM_s105
                                 + 1.3223 w_k1_INTC_s105 + 1.36242 w_k1_KO_s105
                                 + 1.80505 w_k1_MAT_s105
                                 + 1.01366 w_k1_MSFT_s105 + 1.0303 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.946029 w_k1_ADM_s106
                                 + 0.848809 w_k1_AXP_s106
                                 + 1.03604 w_k1_DIS_s106
                                 + 1.03261 w_k1_FDX_s106
                                 + 1.14089 w_k1_IBM_s106
                                 + 0.934404 w_k1_INTC_s106
                                 + 1.07598 w_k1_KO_s106
                                 + 0.828159 w_k1_MAT_s106
                                 + 1.04921 w_k1_MSFT_s106
                                 + 0.972936 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.952798 w_k1_ADM_s106
                                 + 1.21494 w_k1_AXP_s106
                                 + 0.930961 w_k1_DIS_s106
                                 + 1.04965 w_k1_FDX_s106
                                 + 1.21731 w_k1_IBM_s106
                                 + 1.25246 w_k1_INTC_s106
                                 + 0.94296 w_k1_KO_s106 + 1.21696 w_k1_MAT_s106
                                 + 1.11166 w_k1_MSFT_s106
                                 + 0.995604 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.12504 w_k1_ADM_s106
                                 + 1.03516 w_k1_AXP_s106
                                 + 1.39354 w_k1_DIS_s106
                                 + 1.20053 w_k1_FDX_s106
                                 + 0.910002 w_k1_IBM_s106
                                 + 1.11516 w_k1_INTC_s106
                                 + 1.12364 w_k1_KO_s106
                                 + 0.804792 w_k1_MAT_s106
                                 + 1.28437 w_k1_MSFT_s106
                                 + 1.03995 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.32511 w_k1_ADM_s106
                                 + 1.24442 w_k1_AXP_s106
                                 + 1.60896 w_k1_DIS_s106
                                 + 1.28039 w_k1_FDX_s106
                                 + 1.61955 w_k1_IBM_s106
                                 + 1.09839 w_k1_INTC_s106
                                 + 1.37631 w_k1_KO_s106 + 1.32398 w_k1_MAT_s106
                                 + 1.22454 w_k1_MSFT_s106 + 1.0575 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.888312 w_k1_ADM_s106
                                 + 0.713001 w_k1_AXP_s106
                                 + 0.717819 w_k1_DIS_s106
                                 + 0.841707 w_k1_FDX_s106
                                 + 1.11167 w_k1_IBM_s106
                                 + 0.981826 w_k1_INTC_s106
                                 + 0.860914 w_k1_KO_s106
                                 + 0.727433 w_k1_MAT_s106
                                 + 0.904748 w_k1_MSFT_s106
                                 + 0.825949 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.947958 w_k1_ADM_s106
                                 + 0.70894 w_k1_AXP_s106
                                 + 0.711343 w_k1_DIS_s106
                                 + 0.849479 w_k1_FDX_s106
                                 + 0.699011 w_k1_IBM_s106
                                 + 0.796446 w_k1_INTC_s106
                                 + 0.732215 w_k1_KO_s106
                                 + 0.565266 w_k1_MAT_s106
                                 + 0.96506 w_k1_MSFT_s106
                                 + 0.816916 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.08898 w_k1_ADM_s106
                                 + 1.17502 w_k1_AXP_s106
                                 + 0.957072 w_k1_DIS_s106
                                 + 1.09309 w_k1_FDX_s106
                                 + 0.779389 w_k1_IBM_s106
                                 + 1.10564 w_k1_INTC_s106
                                 + 0.911322 w_k1_KO_s106
                                 + 0.970413 w_k1_MAT_s106
                                 + 1.07654 w_k1_MSFT_s106
                                 + 0.986817 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.00023 w_k1_ADM_s106
                                 + 1.44286 w_k1_AXP_s106
                                 + 1.32777 w_k1_DIS_s106
                                 + 1.18939 w_k1_FDX_s106
                                 + 1.15508 w_k1_IBM_s106 + 1.302 w_k1_INTC_s106
                                 + 1.06178 w_k1_KO_s106 + 1.30938 w_k1_MAT_s106
                                 + 1.23558 w_k1_MSFT_s106
                                 + 1.08767 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.931806 w_k1_ADM_s107
                                 + 1.3278 w_k1_AXP_s107
                                 + 0.960521 w_k1_DIS_s107
                                 + 1.16168 w_k1_FDX_s107
                                 + 0.926476 w_k1_IBM_s107
                                 + 0.968499 w_k1_INTC_s107
                                 + 0.84939 w_k1_KO_s107 + 1.09325 w_k1_MAT_s107
                                 + 1.08113 w_k1_MSFT_s107
                                 + 1.01383 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.31096 w_k1_ADM_s107
                                 + 1.72244 w_k1_AXP_s107
                                 + 1.85607 w_k1_DIS_s107
                                 + 1.54787 w_k1_FDX_s107
                                 + 1.20427 w_k1_IBM_s107
                                 + 1.39233 w_k1_INTC_s107
                                 + 1.39632 w_k1_KO_s107 + 1.43385 w_k1_MAT_s107
                                 + 1.61636 w_k1_MSFT_s107
                                 + 1.48148 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.80858 w_k1_ADM_s107
                                 + 0.96442 w_k1_AXP_s107
                                 + 0.808099 w_k1_DIS_s107
                                 + 0.960152 w_k1_FDX_s107
                                 + 0.703706 w_k1_IBM_s107
                                 + 0.760062 w_k1_INTC_s107
                                 + 0.772876 w_k1_KO_s107
                                 + 0.879684 w_k1_MAT_s107
                                 + 0.805351 w_k1_MSFT_s107
                                 + 0.909245 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.795668 w_k1_ADM_s107
                                 + 1.0322 w_k1_AXP_s107
                                 + 0.816503 w_k1_DIS_s107
                                 + 0.821405 w_k1_FDX_s107
                                 + 1.18164 w_k1_IBM_s107
                                 + 0.700138 w_k1_INTC_s107
                                 + 0.978761 w_k1_KO_s107
                                 + 1.51927 w_k1_MAT_s107
                                 + 0.845099 w_k1_MSFT_s107
                                 + 0.93085 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.68029 w_k1_ADM_s107
                                 + 0.794713 w_k1_AXP_s107
                                 + 0.750766 w_k1_DIS_s107
                                 + 0.775017 w_k1_FDX_s107
                                 + 0.928562 w_k1_IBM_s107
                                 + 0.835626 w_k1_INTC_s107
                                 + 0.830074 w_k1_KO_s107
                                 + 0.963508 w_k1_MAT_s107
                                 + 0.88679 w_k1_MSFT_s107
                                 + 0.843527 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.637137 w_k1_ADM_s107
                                 + 0.606636 w_k1_AXP_s107
                                 + 0.678014 w_k1_DIS_s107
                                 + 0.750351 w_k1_FDX_s107
                                 + 0.604598 w_k1_IBM_s107
                                 + 0.851869 w_k1_INTC_s107
                                 + 0.70656 w_k1_KO_s107
                                 + 0.714118 w_k1_MAT_s107
                                 + 0.809138 w_k1_MSFT_s107
                                 + 0.815147 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.18494 w_k1_ADM_s107
                                 + 1.33573 w_k1_AXP_s107
                                 + 1.30239 w_k1_DIS_s107
                                 + 1.15321 w_k1_FDX_s107
                                 + 1.46477 w_k1_IBM_s107
                                 + 1.12327 w_k1_INTC_s107 + 1.5177 w_k1_KO_s107
                                 + 1.76656 w_k1_MAT_s107
                                 + 1.26995 w_k1_MSFT_s107
                                 + 1.18499 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.955701 w_k1_ADM_s107
                                 + 1.14203 w_k1_AXP_s107
                                 + 1.53439 w_k1_DIS_s107
                                 + 1.20321 w_k1_FDX_s107
                                 + 0.978876 w_k1_IBM_s107
                                 + 1.09039 w_k1_INTC_s107
                                 + 1.15076 w_k1_KO_s107
                                 + 0.904394 w_k1_MAT_s107
                                 + 1.21373 w_k1_MSFT_s107
                                 + 1.04367 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.806554 w_k1_ADM_s110
                                 + 0.845382 w_k1_AXP_s110
                                 + 0.608331 w_k1_DIS_s110
                                 + 0.741736 w_k1_FDX_s110
                                 + 1.07982 w_k1_IBM_s110
                                 + 0.73446 w_k1_INTC_s110
                                 + 0.880366 w_k1_KO_s110
                                 + 0.775092 w_k1_MAT_s110
                                 + 0.778659 w_k1_MSFT_s110
                                 + 0.757074 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.925025 w_k1_ADM_s110
                                 + 1.16969 w_k1_AXP_s110
                                 + 0.83016 w_k1_DIS_s110
                                 + 0.936517 w_k1_FDX_s110
                                 + 0.966646 w_k1_IBM_s110
                                 + 0.970071 w_k1_INTC_s110
                                 + 0.910505 w_k1_KO_s110
                                 + 1.01435 w_k1_MAT_s110
                                 + 0.835227 w_k1_MSFT_s110
                                 + 1.01307 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.954306 w_k1_ADM_s110
                                 + 1.09653 w_k1_AXP_s110
                                 + 1.35041 w_k1_DIS_s110
                                 + 1.21893 w_k1_FDX_s110
                                 + 0.866102 w_k1_IBM_s110
                                 + 1.19859 w_k1_INTC_s110
                                 + 1.11125 w_k1_KO_s110
                                 + 0.539894 w_k1_MAT_s110
                                 + 1.24132 w_k1_MSFT_s110
                                 + 1.06239 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.941742 w_k1_ADM_s110
                                 + 0.915027 w_k1_AXP_s110
                                 + 0.956953 w_k1_DIS_s110
                                 + 0.932172 w_k1_FDX_s110
                                 + 0.864059 w_k1_IBM_s110
                                 + 0.951149 w_k1_INTC_s110
                                 + 0.867604 w_k1_KO_s110
                                 + 0.666135 w_k1_MAT_s110
                                 + 0.872784 w_k1_MSFT_s110
                                 + 0.940431 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.945789 w_k1_ADM_s110
                                 + 1.00997 w_k1_AXP_s110
                                 + 1.13945 w_k1_DIS_s110
                                 + 0.94876 w_k1_FDX_s110
                                 + 1.32526 w_k1_IBM_s110
                                 + 0.868175 w_k1_INTC_s110
                                 + 1.16248 w_k1_KO_s110 + 1.00784 w_k1_MAT_s110
                                 + 1.05068 w_k1_MSFT_s110
                                 + 0.994021 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.795548 w_k1_ADM_s110
                                 + 0.712835 w_k1_AXP_s110
                                 + 0.628016 w_k1_DIS_s110
                                 + 0.737506 w_k1_FDX_s110
                                 + 0.666442 w_k1_IBM_s110
                                 + 0.750537 w_k1_INTC_s110
                                 + 0.771846 w_k1_KO_s110
                                 + 0.622557 w_k1_MAT_s110
                                 + 0.74362 w_k1_MSFT_s110
                                 + 0.743851 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.959618 w_k1_ADM_s110
                                 + 0.951769 w_k1_AXP_s110
                                 + 0.729688 w_k1_DIS_s110
                                 + 0.974379 w_k1_FDX_s110
                                 + 0.611473 w_k1_IBM_s110
                                 + 0.931651 w_k1_INTC_s110
                                 + 0.809069 w_k1_KO_s110
                                 + 0.325154 w_k1_MAT_s110
                                 + 0.934098 w_k1_MSFT_s110
                                 + 0.839008 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.24824 w_k1_ADM_s110
                                 + 1.5105 w_k1_AXP_s110 + 1.36706 w_k1_DIS_s110
                                 + 1.2803 w_k1_FDX_s110 + 1.25567 w_k1_IBM_s110
                                 + 0.98738 w_k1_INTC_s110 + 1.2401 w_k1_KO_s110
                                 + 0.957093 w_k1_MAT_s110
                                 + 1.09297 w_k1_MSFT_s110
                                 + 1.09998 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.03279 w_k1_ADM_s111
                                 + 1.05241 w_k1_AXP_s111
                                 + 0.936047 w_k1_DIS_s111
                                 + 1.05202 w_k1_FDX_s111
                                 + 0.809574 w_k1_IBM_s111
                                 + 1.14417 w_k1_INTC_s111
                                 + 0.799101 w_k1_KO_s111
                                 + 0.875111 w_k1_MAT_s111
                                 + 0.99101 w_k1_MSFT_s111
                                 + 0.982989 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.32135 w_k1_ADM_s111
                                 + 1.36869 w_k1_AXP_s111
                                 + 1.57221 w_k1_DIS_s111
                                 + 1.30907 w_k1_FDX_s111
                                 + 1.32792 w_k1_IBM_s111
                                 + 1.41011 w_k1_INTC_s111
                                 + 1.32613 w_k1_KO_s111 + 1.10851 w_k1_MAT_s111
                                 + 1.34021 w_k1_MSFT_s111
                                 + 1.21744 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.848254 w_k1_ADM_s111
                                 + 0.989806 w_k1_AXP_s111
                                 + 0.934222 w_k1_DIS_s111
                                 + 1.04525 w_k1_FDX_s111
                                 + 0.756202 w_k1_IBM_s111
                                 + 1.11084 w_k1_INTC_s111
                                 + 0.742461 w_k1_KO_s111
                                 + 0.740296 w_k1_MAT_s111
                                 + 1.16239 w_k1_MSFT_s111
                                 + 0.855634 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.882308 w_k1_ADM_s111
                                 + 1.0459 w_k1_AXP_s111
                                 + 0.870255 w_k1_DIS_s111
                                 + 0.830172 w_k1_FDX_s111
                                 + 1.45405 w_k1_IBM_s111
                                 + 1.11763 w_k1_INTC_s111
                                 + 0.974412 w_k1_KO_s111
                                 + 0.993581 w_k1_MAT_s111
                                 + 0.97127 w_k1_MSFT_s111 + 1.0473 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03513 w_k1_ADM_s111
                                 + 1.35808 w_k1_AXP_s111
                                 + 1.52294 w_k1_DIS_s111
                                 + 1.36788 w_k1_FDX_s111
                                 + 0.847556 w_k1_IBM_s111
                                 + 1.5033 w_k1_INTC_s111 + 1.08593 w_k1_KO_s111
                                 + 0.7058 w_k1_MAT_s111 + 1.3797 w_k1_MSFT_s111
                                 + 1.06203 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.841209 w_k1_ADM_s111
                                 + 1.20505 w_k1_AXP_s111
                                 + 1.25049 w_k1_DIS_s111
                                 + 1.03634 w_k1_FDX_s111
                                 + 1.42686 w_k1_IBM_s111
                                 + 1.28986 w_k1_INTC_s111
                                 + 1.02936 w_k1_KO_s111 + 1.35666 w_k1_MAT_s111
                                 + 1.15944 w_k1_MSFT_s111
                                 + 1.01244 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.12497 w_k1_ADM_s111
                                 + 1.11189 w_k1_AXP_s111
                                 + 1.06702 w_k1_DIS_s111
                                 + 1.05833 w_k1_FDX_s111
                                 + 1.04055 w_k1_IBM_s111
                                 + 1.32234 w_k1_INTC_s111
                                 + 0.874214 w_k1_KO_s111
                                 + 1.03503 w_k1_MAT_s111
                                 + 1.11811 w_k1_MSFT_s111
                                 + 0.992578 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.794309 w_k1_ADM_s111
                                 + 0.687626 w_k1_AXP_s111
                                 + 0.583777 w_k1_DIS_s111
                                 + 0.714028 w_k1_FDX_s111
                                 + 0.88511 w_k1_IBM_s111
                                 + 0.994238 w_k1_INTC_s111
                                 + 0.763749 w_k1_KO_s111
                                 + 0.653505 w_k1_MAT_s111
                                 + 0.798597 w_k1_MSFT_s111
                                 + 0.816394 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.12815 w_k1_ADM_s112
                                 + 1.17971 w_k1_AXP_s112
                                 + 1.01498 w_k1_DIS_s112
                                 + 1.08835 w_k1_FDX_s112 + 1.0687 w_k1_IBM_s112
                                 + 1.02963 w_k1_INTC_s112
                                 + 1.24914 w_k1_KO_s112
                                 + 0.949879 w_k1_MAT_s112
                                 + 0.940349 w_k1_MSFT_s112
                                 + 1.05613 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.0456 w_k1_ADM_s112
                                 + 1.78382 w_k1_AXP_s112 + 1.6132 w_k1_DIS_s112
                                 + 1.4419 w_k1_FDX_s112 + 1.32877 w_k1_IBM_s112
                                 + 1.64803 w_k1_INTC_s112
                                 + 1.35964 w_k1_KO_s112 + 1.44394 w_k1_MAT_s112
                                 + 1.23154 w_k1_MSFT_s112
                                 + 1.14128 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.985184 w_k1_ADM_s112
                                 + 0.885386 w_k1_AXP_s112
                                 + 0.930492 w_k1_DIS_s112
                                 + 0.969864 w_k1_FDX_s112
                                 + 0.876266 w_k1_IBM_s112
                                 + 0.965602 w_k1_INTC_s112
                                 + 1.01775 w_k1_KO_s112
                                 + 0.962076 w_k1_MAT_s112
                                 + 0.949907 w_k1_MSFT_s112
                                 + 0.922625 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.28415 w_k1_ADM_s112
                                 + 1.17635 w_k1_AXP_s112
                                 + 1.41863 w_k1_DIS_s112
                                 + 1.09589 w_k1_FDX_s112
                                 + 1.42288 w_k1_IBM_s112
                                 + 1.00057 w_k1_INTC_s112
                                 + 1.40743 w_k1_KO_s112 + 1.278 w_k1_MAT_s112
                                 + 1.15268 w_k1_MSFT_s112
                                 + 1.19875 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.811756 w_k1_ADM_s112
                                 + 0.88353 w_k1_AXP_s112
                                 + 0.670846 w_k1_DIS_s112
                                 + 0.779938 w_k1_FDX_s112
                                 + 1.2078 w_k1_IBM_s112
                                 + 0.918856 w_k1_INTC_s112
                                 + 1.06467 w_k1_KO_s112 + 1.14027 w_k1_MAT_s112
                                 + 0.703179 w_k1_MSFT_s112
                                 + 0.903468 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.970489 w_k1_ADM_s112
                                 + 1.40117 w_k1_AXP_s112
                                 + 0.814755 w_k1_DIS_s112
                                 + 0.934409 w_k1_FDX_s112
                                 + 1.17765 w_k1_IBM_s112
                                 + 0.946802 w_k1_INTC_s112
                                 + 1.09754 w_k1_KO_s112 + 1.55195 w_k1_MAT_s112
                                 + 0.878267 w_k1_MSFT_s112
                                 + 0.994488 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.13743 w_k1_ADM_s112
                                 + 1.34142 w_k1_AXP_s112
                                 + 1.51083 w_k1_DIS_s112
                                 + 1.37413 w_k1_FDX_s112
                                 + 0.882415 w_k1_IBM_s112
                                 + 1.2262 w_k1_INTC_s112 + 1.32591 w_k1_KO_s112
                                 + 0.913919 w_k1_MAT_s112
                                 + 1.10309 w_k1_MSFT_s112
                                 + 1.12359 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.871679 w_k1_ADM_s112
                                 + 0.713766 w_k1_AXP_s112
                                 + 0.604025 w_k1_DIS_s112
                                 + 0.751126 w_k1_FDX_s112
                                 + 0.693574 w_k1_IBM_s112
                                 + 0.844887 w_k1_INTC_s112
                                 + 0.901998 w_k1_KO_s112
                                 + 0.61048 w_k1_MAT_s112
                                 + 0.74287 w_k1_MSFT_s112
                                 + 0.822368 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.04153 w_k1_ADM_s113
                                 + 1.10979 w_k1_AXP_s113
                                 + 1.34419 w_k1_DIS_s113
                                 + 1.08885 w_k1_FDX_s113
                                 + 1.51248 w_k1_IBM_s113
                                 + 1.21574 w_k1_INTC_s113
                                 + 1.10912 w_k1_KO_s113 + 1.51009 w_k1_MAT_s113
                                 + 1.24821 w_k1_MSFT_s113 + 1.1062 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.986683 w_k1_ADM_s113
                                 + 1.01954 w_k1_AXP_s113
                                 + 0.917446 w_k1_DIS_s113
                                 + 1.01024 w_k1_FDX_s113
                                 + 0.927018 w_k1_IBM_s113
                                 + 0.983443 w_k1_INTC_s113
                                 + 0.882864 w_k1_KO_s113
                                 + 0.752067 w_k1_MAT_s113
                                 + 0.983727 w_k1_MSFT_s113
                                 + 1.00337 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.4474 w_k1_ADM_s113
                                 + 1.45843 w_k1_AXP_s113
                                 + 1.79582 w_k1_DIS_s113
                                 + 1.53434 w_k1_FDX_s113
                                 + 1.14437 w_k1_IBM_s113
                                 + 1.42203 w_k1_INTC_s113 + 1.0101 w_k1_KO_s113
                                 + 1.21313 w_k1_MAT_s113
                                 + 1.54261 w_k1_MSFT_s113
                                 + 1.18846 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.10236 w_k1_ADM_s113
                                 + 1.23151 w_k1_AXP_s113
                                 + 1.12602 w_k1_DIS_s113
                                 + 1.21565 w_k1_FDX_s113
                                 + 0.898596 w_k1_IBM_s113
                                 + 1.23541 w_k1_INTC_s113
                                 + 0.837563 w_k1_KO_s113
                                 + 1.09227 w_k1_MAT_s113
                                 + 1.18601 w_k1_MSFT_s113
                                 + 1.02491 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.84192 w_k1_ADM_s113
                                 + 0.758919 w_k1_AXP_s113
                                 + 0.746194 w_k1_DIS_s113
                                 + 0.763823 w_k1_FDX_s113
                                 + 0.954413 w_k1_IBM_s113
                                 + 0.804613 w_k1_INTC_s113
                                 + 0.713566 w_k1_KO_s113
                                 + 0.898931 w_k1_MAT_s113
                                 + 0.808115 w_k1_MSFT_s113
                                 + 0.875422 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.3014 w_k1_ADM_s113
                                 + 1.04402 w_k1_AXP_s113
                                 + 0.70751 w_k1_DIS_s113
                                 + 1.01855 w_k1_FDX_s113
                                 + 1.05957 w_k1_IBM_s113
                                 + 0.840462 w_k1_INTC_s113
                                 + 0.882074 w_k1_KO_s113
                                 + 0.980129 w_k1_MAT_s113
                                 + 1.03122 w_k1_MSFT_s113
                                 + 0.988867 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05139 w_k1_ADM_s113
                                 + 1.29707 w_k1_AXP_s113
                                 + 0.913766 w_k1_DIS_s113
                                 + 0.987631 w_k1_FDX_s113
                                 + 1.32227 w_k1_IBM_s113
                                 + 0.981628 w_k1_INTC_s113
                                 + 0.965421 w_k1_KO_s113
                                 + 1.79136 w_k1_MAT_s113
                                 + 1.00649 w_k1_MSFT_s113
                                 + 0.92279 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.04737 w_k1_ADM_s113
                                 + 0.760969 w_k1_AXP_s113
                                 + 0.846853 w_k1_DIS_s113
                                 + 0.877729 w_k1_FDX_s113
                                 + 0.632476 w_k1_IBM_s113
                                 + 0.917206 w_k1_INTC_s113
                                 + 0.6945 w_k1_KO_s113 + 0.572884 w_k1_MAT_s113
                                 + 1.04855 w_k1_MSFT_s113
                                 + 0.786244 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.827239 w_k1_ADM_s114
                                 + 0.826122 w_k1_AXP_s114
                                 + 0.661151 w_k1_DIS_s114
                                 + 0.914804 w_k1_FDX_s114
                                 + 0.615908 w_k1_IBM_s114
                                 + 0.880063 w_k1_INTC_s114
                                 + 0.689479 w_k1_KO_s114
                                 + 0.922046 w_k1_MAT_s114
                                 + 0.833561 w_k1_MSFT_s114
                                 + 0.859342 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.979382 w_k1_ADM_s114
                                 + 0.975721 w_k1_AXP_s114
                                 + 0.888512 w_k1_DIS_s114
                                 + 0.898458 w_k1_FDX_s114
                                 + 1.51455 w_k1_IBM_s114
                                 + 0.925545 w_k1_INTC_s114
                                 + 1.18847 w_k1_KO_s114 + 1.81337 w_k1_MAT_s114
                                 + 1.05624 w_k1_MSFT_s114
                                 + 1.14179 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.848766 w_k1_ADM_s114
                                 + 0.767712 w_k1_AXP_s114
                                 + 0.659818 w_k1_DIS_s114
                                 + 0.76034 w_k1_FDX_s114
                                 + 0.944259 w_k1_IBM_s114
                                 + 0.636071 w_k1_INTC_s114
                                 + 0.851738 w_k1_KO_s114
                                 + 0.956912 w_k1_MAT_s114
                                 + 0.746872 w_k1_MSFT_s114
                                 + 0.926131 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.11242 w_k1_ADM_s114
                                 + 1.3782 w_k1_AXP_s114 + 1.26812 w_k1_DIS_s114
                                 + 1.16404 w_k1_FDX_s114 + 1.3127 w_k1_IBM_s114
                                 + 1.06882 w_k1_INTC_s114
                                 + 1.21461 w_k1_KO_s114 + 1.72821 w_k1_MAT_s114
                                 + 1.27181 w_k1_MSFT_s114
                                 + 1.13038 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.12474 w_k1_ADM_s114
                                 + 1.24061 w_k1_AXP_s114
                                 + 1.48175 w_k1_DIS_s114
                                 + 1.22507 w_k1_FDX_s114
                                 + 1.15536 w_k1_IBM_s114
                                 + 0.927845 w_k1_INTC_s114
                                 + 1.35946 w_k1_KO_s114 + 1.4404 w_k1_MAT_s114
                                 + 1.21065 w_k1_MSFT_s114
                                 + 1.37834 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.99259 w_k1_ADM_s114
                                 + 0.914572 w_k1_AXP_s114
                                 + 1.13194 w_k1_DIS_s114
                                 + 1.05937 w_k1_FDX_s114
                                 + 0.806757 w_k1_IBM_s114
                                 + 0.766825 w_k1_INTC_s114
                                 + 1.03573 w_k1_KO_s114 + 1.08148 w_k1_MAT_s114
                                 + 1.10964 w_k1_MSFT_s114
                                 + 1.08513 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.93736 w_k1_ADM_s114
                                 + 1.01 w_k1_AXP_s114 + 0.863975 w_k1_DIS_s114
                                 + 0.874744 w_k1_FDX_s114
                                 + 1.12379 w_k1_IBM_s114
                                 + 0.85279 w_k1_INTC_s114
                                 + 0.976885 w_k1_KO_s114
                                 + 1.28973 w_k1_MAT_s114
                                 + 0.922091 w_k1_MSFT_s114
                                 + 0.976148 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.988322 w_k1_ADM_s114
                                 + 1.62232 w_k1_AXP_s114
                                 + 1.29999 w_k1_DIS_s114
                                 + 1.36746 w_k1_FDX_s114
                                 + 0.88683 w_k1_IBM_s114
                                 + 1.22735 w_k1_INTC_s114
                                 + 1.20492 w_k1_KO_s114 + 1.45687 w_k1_MAT_s114
                                 + 1.33562 w_k1_MSFT_s114
                                 + 1.29739 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.988778 w_k1_ADM_s115
                                 + 1.00272 w_k1_AXP_s115
                                 + 1.00528 w_k1_DIS_s115
                                 + 0.952795 w_k1_FDX_s115
                                 + 1.01782 w_k1_IBM_s115
                                 + 0.98982 w_k1_INTC_s115
                                 + 1.07263 w_k1_KO_s115 + 1.11769 w_k1_MAT_s115
                                 + 1.13148 w_k1_MSFT_s115
                                 + 1.08115 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.886517 w_k1_ADM_s115
                                 + 0.917055 w_k1_AXP_s115
                                 + 1.01228 w_k1_DIS_s115
                                 + 0.950399 w_k1_FDX_s115
                                 + 0.791888 w_k1_IBM_s115
                                 + 1.03295 w_k1_INTC_s115
                                 + 0.912154 w_k1_KO_s115
                                 + 0.77178 w_k1_MAT_s115
                                 + 0.904724 w_k1_MSFT_s115
                                 + 0.962187 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.844677 w_k1_ADM_s115
                                 + 0.672529 w_k1_AXP_s115
                                 + 0.68242 w_k1_DIS_s115
                                 + 0.694118 w_k1_FDX_s115
                                 + 0.985764 w_k1_IBM_s115
                                 + 0.782554 w_k1_INTC_s115
                                 + 0.817925 w_k1_KO_s115
                                 + 0.607759 w_k1_MAT_s115
                                 + 0.739406 w_k1_MSFT_s115
                                 + 0.822084 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.9971 w_k1_ADM_s115
                                 + 1.15291 w_k1_AXP_s115
                                 + 1.19551 w_k1_DIS_s115
                                 + 0.928878 w_k1_FDX_s115
                                 + 1.50909 w_k1_IBM_s115
                                 + 0.952927 w_k1_INTC_s115
                                 + 1.37804 w_k1_KO_s115 + 1.44392 w_k1_MAT_s115
                                 + 1.0358 w_k1_MSFT_s115 + 1.08093 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.976893 w_k1_ADM_s115
                                 + 1.00193 w_k1_AXP_s115
                                 + 0.69586 w_k1_DIS_s115
                                 + 0.794562 w_k1_FDX_s115
                                 + 1.17821 w_k1_IBM_s115
                                 + 0.820152 w_k1_INTC_s115
                                 + 0.944526 w_k1_KO_s115
                                 + 1.41581 w_k1_MAT_s115
                                 + 0.750559 w_k1_MSFT_s115
                                 + 0.996812 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.837286 w_k1_ADM_s115
                                 + 0.991753 w_k1_AXP_s115
                                 + 1.05437 w_k1_DIS_s115
                                 + 0.925661 w_k1_FDX_s115
                                 + 1.21797 w_k1_IBM_s115
                                 + 1.10767 w_k1_INTC_s115
                                 + 1.15626 w_k1_KO_s115 + 1.31436 w_k1_MAT_s115
                                 + 1.0907 w_k1_MSFT_s115 + 1.10313 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.24472 w_k1_ADM_s115
                                 + 1.28045 w_k1_AXP_s115
                                 + 1.44124 w_k1_DIS_s115
                                 + 1.26717 w_k1_FDX_s115
                                 + 0.811328 w_k1_IBM_s115
                                 + 1.22226 w_k1_INTC_s115
                                 + 1.12504 w_k1_KO_s115
                                 + 0.878034 w_k1_MAT_s115
                                 + 1.19492 w_k1_MSFT_s115
                                 + 1.14989 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.996387 w_k1_ADM_s115
                                 + 1.6657 w_k1_AXP_s115 + 1.60499 w_k1_DIS_s115
                                 + 1.26288 w_k1_FDX_s115
                                 + 1.55991 w_k1_IBM_s115
                                 + 1.45652 w_k1_INTC_s115
                                 + 1.38687 w_k1_KO_s115 + 2.13852 w_k1_MAT_s115
                                 + 1.34037 w_k1_MSFT_s115
                                 + 1.32252 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.18343 w_k1_ADM_s116
                                 + 0.84757 w_k1_AXP_s116
                                 + 0.760385 w_k1_DIS_s116 + 1.015 w_k1_FDX_s116
                                 + 0.632608 w_k1_IBM_s116
                                 + 0.962926 w_k1_INTC_s116
                                 + 0.733021 w_k1_KO_s116
                                 + 0.742863 w_k1_MAT_s116
                                 + 0.939762 w_k1_MSFT_s116
                                 + 0.817419 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.21267 w_k1_ADM_s116
                                 + 0.875733 w_k1_AXP_s116
                                 + 0.870992 w_k1_DIS_s116
                                 + 0.890066 w_k1_FDX_s116
                                 + 0.966009 w_k1_IBM_s116
                                 + 0.735742 w_k1_INTC_s116
                                 + 0.966858 w_k1_KO_s116
                                 + 1.31354 w_k1_MAT_s116
                                 + 0.95659 w_k1_MSFT_s116
                                 + 0.957085 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.2695 w_k1_ADM_s116
                                 + 1.22434 w_k1_AXP_s116
                                 + 1.23978 w_k1_DIS_s116
                                 + 1.21196 w_k1_FDX_s116
                                 + 0.746314 w_k1_IBM_s116
                                 + 1.28325 w_k1_INTC_s116
                                 + 0.935391 w_k1_KO_s116
                                 + 1.15502 w_k1_MAT_s116
                                 + 1.3628 w_k1_MSFT_s116 + 1.10165 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.895319 w_k1_ADM_s116
                                 + 1.03343 w_k1_AXP_s116
                                 + 0.976776 w_k1_DIS_s116
                                 + 1.04614 w_k1_FDX_s116
                                 + 0.933107 w_k1_IBM_s116
                                 + 0.889129 w_k1_INTC_s116
                                 + 0.952061 w_k1_KO_s116
                                 + 1.31821 w_k1_MAT_s116
                                 + 1.15931 w_k1_MSFT_s116
                                 + 1.00925 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.916928 w_k1_ADM_s116
                                 + 1.10227 w_k1_AXP_s116
                                 + 1.30993 w_k1_DIS_s116
                                 + 1.12355 w_k1_FDX_s116
                                 + 1.40653 w_k1_IBM_s116
                                 + 1.1568 w_k1_INTC_s116 + 1.3194 w_k1_KO_s116
                                 + 1.40696 w_k1_MAT_s116
                                 + 1.18111 w_k1_MSFT_s116
                                 + 1.12451 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.783538 w_k1_ADM_s116
                                 + 0.599051 w_k1_AXP_s116
                                 + 0.600364 w_k1_DIS_s116
                                 + 0.777478 w_k1_FDX_s116
                                 + 0.706865 w_k1_IBM_s116
                                 + 0.798602 w_k1_INTC_s116
                                 + 0.728708 w_k1_KO_s116
                                 + 0.623033 w_k1_MAT_s116
                                 + 0.858646 w_k1_MSFT_s116
                                 + 0.788352 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.845314 w_k1_ADM_s116
                                 + 0.874065 w_k1_AXP_s116
                                 + 0.904249 w_k1_DIS_s116
                                 + 0.853832 w_k1_FDX_s116
                                 + 1.28066 w_k1_IBM_s116
                                 + 0.831441 w_k1_INTC_s116
                                 + 1.1979 w_k1_KO_s116 + 1.69105 w_k1_MAT_s116
                                 + 1.02052 w_k1_MSFT_s116
                                 + 1.00998 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.24544 w_k1_ADM_s116
                                 + 1.3883 w_k1_AXP_s116 + 1.3804 w_k1_DIS_s116
                                 + 1.19074 w_k1_FDX_s116
                                 + 1.25624 w_k1_IBM_s116
                                 + 1.3116 w_k1_INTC_s116 + 1.28123 w_k1_KO_s116
                                 + 1.47096 w_k1_MAT_s116
                                 + 1.2842 w_k1_MSFT_s116 + 1.11069 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.940397 w_k1_ADM_s117
                                 + 1.10342 w_k1_AXP_s117
                                 + 1.02052 w_k1_DIS_s117
                                 + 1.06413 w_k1_FDX_s117
                                 + 0.896294 w_k1_IBM_s117
                                 + 0.857123 w_k1_INTC_s117
                                 + 0.879188 w_k1_KO_s117
                                 + 1.31036 w_k1_MAT_s117
                                 + 0.948824 w_k1_MSFT_s117
                                 + 1.04063 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.00632 w_k1_ADM_s117
                                 + 0.822831 w_k1_AXP_s117
                                 + 0.98869 w_k1_DIS_s117
                                 + 0.907025 w_k1_FDX_s117
                                 + 1.05638 w_k1_IBM_s117
                                 + 0.865163 w_k1_INTC_s117
                                 + 1.06113 w_k1_KO_s117
                                 + 0.894988 w_k1_MAT_s117
                                 + 1.0535 w_k1_MSFT_s117 + 1.03541 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.866115 w_k1_ADM_s117
                                 + 0.849806 w_k1_AXP_s117
                                 + 0.790025 w_k1_DIS_s117
                                 + 0.877736 w_k1_FDX_s117
                                 + 0.771275 w_k1_IBM_s117
                                 + 0.704038 w_k1_INTC_s117
                                 + 0.853991 w_k1_KO_s117
                                 + 0.963965 w_k1_MAT_s117
                                 + 0.934927 w_k1_MSFT_s117
                                 + 0.898796 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.81049 w_k1_ADM_s117
                                 + 0.940503 w_k1_AXP_s117
                                 + 1.10112 w_k1_DIS_s117
                                 + 0.928646 w_k1_FDX_s117
                                 + 1.55898 w_k1_IBM_s117
                                 + 0.73475 w_k1_INTC_s117
                                 + 1.27861 w_k1_KO_s117 + 1.61817 w_k1_MAT_s117
                                 + 0.950898 w_k1_MSFT_s117
                                 + 1.09108 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.3362 w_k1_ADM_s117
                                 + 1.09188 w_k1_AXP_s117
                                 + 1.38189 w_k1_DIS_s117
                                 + 1.14297 w_k1_FDX_s117 + 1.331 w_k1_IBM_s117
                                 + 0.996973 w_k1_INTC_s117
                                 + 1.5597 w_k1_KO_s117 + 1.56929 w_k1_MAT_s117
                                 + 1.26724 w_k1_MSFT_s117
                                 + 1.14738 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.760238 w_k1_ADM_s117
                                 + 0.985993 w_k1_AXP_s117
                                 + 0.762267 w_k1_DIS_s117
                                 + 0.80506 w_k1_FDX_s117
                                 + 1.09547 w_k1_IBM_s117
                                 + 0.839123 w_k1_INTC_s117
                                 + 0.91693 w_k1_KO_s117 + 1.35765 w_k1_MAT_s117
                                 + 0.943246 w_k1_MSFT_s117
                                 + 0.967153 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.24771 w_k1_ADM_s117
                                 + 1.43139 w_k1_AXP_s117
                                 + 1.43602 w_k1_DIS_s117
                                 + 1.32152 w_k1_FDX_s117
                                 + 1.19034 w_k1_IBM_s117
                                 + 1.05684 w_k1_INTC_s117 + 1.2404 w_k1_KO_s117
                                 + 1.58938 w_k1_MAT_s117
                                 + 1.05681 w_k1_MSFT_s117
                                 + 1.10703 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.19544 w_k1_ADM_s117
                                 + 1.13183 w_k1_AXP_s117
                                 + 1.27442 w_k1_DIS_s117
                                 + 1.18709 w_k1_FDX_s117
                                 + 0.829954 w_k1_IBM_s117
                                 + 0.859913 w_k1_INTC_s117
                                 + 1.17854 w_k1_KO_s117
                                 + 0.921198 w_k1_MAT_s117
                                 + 1.18844 w_k1_MSFT_s117
                                 + 1.10481 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0149605270848093 w_k1_ADM_s000
                                 + 0.00870719963681893 w_k1_ADM_s001
                                 + 0.0133282380536613 w_k1_ADM_s002
                                 + 0.00760185985835063 w_k1_ADM_s003
                                 + 0.0128511696884249 w_k1_ADM_s004
                                 + 0.0143937019769746 w_k1_ADM_s005
                                 + 0.0252199271871831 w_k1_ADM_s006
                                 + 0.0156347670553749 w_k1_ADM_s007
                                 + 0.0213435258501891 w_k1_ADM_s010
                                 + 0.0134303769099572 w_k1_ADM_s011
                                 + 0.0059027319117722 w_k1_ADM_s012
                                 + 0.0149282920194767 w_k1_ADM_s013
                                 + 0.0168705657003565 w_k1_ADM_s014
                                 + 0.0145463344488094 w_k1_ADM_s015
                                 + 0.00924552855779615 w_k1_ADM_s016
                                 + 0.00476206816470824 w_k1_ADM_s017
                                 + 0.0531072596947095 w_k1_ADM_s100
                                 + 0.0236776388562693 w_k1_ADM_s101
                                 + 0.0311523252131429 w_k1_ADM_s102
                                 + 0.0204301962033939 w_k1_ADM_s103
                                 + 0.029745849591465 w_k1_ADM_s104
                                 + 0.0292538889352108 w_k1_ADM_s105
                                 + 0.0765080697312376 w_k1_ADM_s106
                                 + 0.0281494072487583 w_k1_ADM_s107
                                 + 0.0492473433148937 w_k1_ADM_s110
                                 + 0.0775966532181416 w_k1_ADM_s111
                                 + 0.0723847851603334 w_k1_ADM_s112
                                 + 0.0778860885539127 w_k1_ADM_s113
                                 + 0.0396973162315848 w_k1_ADM_s114
                                 + 0.074704850752039 w_k1_ADM_s115
                                 + 0.0791439213852088 w_k1_ADM_s116
                                 + 0.0299249884258796 w_k1_ADM_s117
                                 + 0.0138941803661414 w_k1_AXP_s000
                                 + 0.00986686247291808 w_k1_AXP_s001
                                 + 0.0143770674852638 w_k1_AXP_s002
                                 + 0.00728737007643089 w_k1_AXP_s003
                                 + 0.0116055622537259 w_k1_AXP_s004
                                 + 0.0120361221167665 w_k1_AXP_s005
                                 + 0.0240721691635672 w_k1_AXP_s006
                                 + 0.0144054173053087 w_k1_AXP_s007
                                 + 0.0242227881427702 w_k1_AXP_s010
                                 + 0.0130323693856428 w_k1_AXP_s011
                                 + 0.00603687916972567 w_k1_AXP_s012
                                 + 0.01632750377921 w_k1_AXP_s013
                                 + 0.0180070082243056 w_k1_AXP_s014
                                 + 0.0154288928199091 w_k1_AXP_s015
                                 + 0.0105612914272003 w_k1_AXP_s016
                                 + 0.00523555472764894 w_k1_AXP_s017
                                 + 0.0508191552374357 w_k1_AXP_s100
                                 + 0.0242731488362645 w_k1_AXP_s101
                                 + 0.0318216448338859 w_k1_AXP_s102
                                 + 0.0189897729286732 w_k1_AXP_s103
                                 + 0.0319386905278552 w_k1_AXP_s104
                                 + 0.0244241409931882 w_k1_AXP_s105
                                 + 0.0739102452445433 w_k1_AXP_s106
                                 + 0.0354617271966532 w_k1_AXP_s107
                                 + 0.0540616672036172 w_k1_AXP_s110
                                 + 0.0843997128926361 w_k1_AXP_s111
                                 + 0.0748489136707604 w_k1_AXP_s112
                                 + 0.0758758449165397 w_k1_AXP_s113
                                 + 0.0433062085606909 w_k1_AXP_s114
                                 + 0.0820936480718986 w_k1_AXP_s115
                                 + 0.0762119579964732 w_k1_AXP_s116
                                 + 0.0307548448541028 w_k1_AXP_s117
                                 + 0.0158757468931135 w_k1_DIS_s000
                                 + 0.0111163929644982 w_k1_DIS_s001
                                 + 0.0149864180009665 w_k1_DIS_s002
                                 + 0.0078072855402344 w_k1_DIS_s003
                                 + 0.0127652435155228 w_k1_DIS_s004
                                 + 0.0135724280643396 w_k1_DIS_s005
                                 + 0.0240104308367013 w_k1_DIS_s006
                                 + 0.0148872625812746 w_k1_DIS_s007
                                 + 0.0240857805695976 w_k1_DIS_s010
                                 + 0.0128866720764932 w_k1_DIS_s011
                                 + 0.00608590429610293 w_k1_DIS_s012
                                 + 0.0152548139864179 w_k1_DIS_s013
                                 + 0.0169927227720821 w_k1_DIS_s014
                                 + 0.0140240613589955 w_k1_DIS_s015
                                 + 0.0114677631452465 w_k1_DIS_s016
                                 + 0.00523614254292239 w_k1_DIS_s017
                                 + 0.0520136449783161 w_k1_DIS_s100
                                 + 0.0245229611519924 w_k1_DIS_s101
                                 + 0.0333443276895262 w_k1_DIS_s102
                                 + 0.0206165812487552 w_k1_DIS_s103
                                 + 0.0320337019359656 w_k1_DIS_s104
                                 + 0.0233478231235286 w_k1_DIS_s105
                                 + 0.0762044630164316 w_k1_DIS_s106
                                 + 0.0318853078783831 w_k1_DIS_s107
                                 + 0.0521108128930763 w_k1_DIS_s110
                                 + 0.0825213342478326 w_k1_DIS_s111
                                 + 0.0712369310739459 w_k1_DIS_s112
                                 + 0.077154487246397 w_k1_DIS_s113
                                 + 0.0411416616326706 w_k1_DIS_s114
                                 + 0.0849134374353855 w_k1_DIS_s115
                                 + 0.0769904271528587 w_k1_DIS_s116
                                 + 0.0311402937035524 w_k1_DIS_s117
                                 + 0.0141945348278142 w_k1_FDX_s000
                                 + 0.00950087625705492 w_k1_FDX_s001
                                 + 0.0146856138887295 w_k1_FDX_s002
                                 + 0.00771762811975741 w_k1_FDX_s003
                                 + 0.0120052510362497 w_k1_FDX_s004
                                 + 0.0127920548059161 w_k1_FDX_s005
                                 + 0.0234807310983063 w_k1_FDX_s006
                                 + 0.014603036173447 w_k1_FDX_s007
                                 + 0.0229676607486246 w_k1_FDX_s010
                                 + 0.0133322306442043 w_k1_FDX_s011
                                 + 0.00585080235904345 w_k1_FDX_s012
                                 + 0.0151868252471286 w_k1_FDX_s013
                                 + 0.0165770286226384 w_k1_FDX_s014
                                 + 0.0142392814879051 w_k1_FDX_s015
                                 + 0.0113885781306584 w_k1_FDX_s016
                                 + 0.005128899723253 w_k1_FDX_s017
                                 + 0.0548463540290546 w_k1_FDX_s100
                                 + 0.02516143629993 w_k1_FDX_s101
                                 + 0.0320507147007613 w_k1_FDX_s102
                                 + 0.0192981084799429 w_k1_FDX_s103
                                 + 0.0331964857875078 w_k1_FDX_s104
                                 + 0.0237288691805904 w_k1_FDX_s105
                                 + 0.0773283167381637 w_k1_FDX_s106
                                 + 0.0323429845506768 w_k1_FDX_s107
                                 + 0.0505268679826521 w_k1_FDX_s110
                                 + 0.0812131477880793 w_k1_FDX_s111
                                 + 0.0710508619073878 w_k1_FDX_s112
                                 + 0.0760336916159455 w_k1_FDX_s113
                                 + 0.0415265452639018 w_k1_FDX_s114
                                 + 0.0760639784125515 w_k1_FDX_s115
                                 + 0.0762032338417879 w_k1_FDX_s116
                                 + 0.0300756225521348 w_k1_FDX_s117
                                 + 0.0147996906956295 w_k1_IBM_s000
                                 + 0.00935656247065086 w_k1_IBM_s001
                                 + 0.0156558270563297 w_k1_IBM_s002
                                 + 0.0081328525894008 w_k1_IBM_s003
                                 + 0.0125467853170218 w_k1_IBM_s004
                                 + 0.0133529881220336 w_k1_IBM_s005
                                 + 0.0254553173109251 w_k1_IBM_s006
                                 + 0.0154525751978355 w_k1_IBM_s007
                                 + 0.0235271192975668 w_k1_IBM_s010
                                 + 0.0139957211391543 w_k1_IBM_s011
                                 + 0.00629077247483518 w_k1_IBM_s012
                                 + 0.0162989639189194 w_k1_IBM_s013
                                 + 0.0163497923973145 w_k1_IBM_s014
                                 + 0.0137754577456131 w_k1_IBM_s015
                                 + 0.0114995183158254 w_k1_IBM_s016
                                 + 0.005067727716876 w_k1_IBM_s017
                                 + 0.0554427093163291 w_k1_IBM_s100
                                 + 0.0239190020508203 w_k1_IBM_s101
                                 + 0.0333309688796699 w_k1_IBM_s102
                                 + 0.0185974860218955 w_k1_IBM_s103
                                 + 0.0339018555066627 w_k1_IBM_s104
                                 + 0.0228729585506657 w_k1_IBM_s105
                                 + 0.0763080148767316 w_k1_IBM_s106
                                 + 0.0315422432045505 w_k1_IBM_s107
                                 + 0.0525169395536013 w_k1_IBM_s110
                                 + 0.0833846202668297 w_k1_IBM_s111
                                 + 0.0733496724538101 w_k1_IBM_s112
                                 + 0.0748079463808384 w_k1_IBM_s113
                                 + 0.0424600886224682 w_k1_IBM_s114
                                 + 0.0794549242545301 w_k1_IBM_s115
                                 + 0.0770947408417121 w_k1_IBM_s116
                                 + 0.0313907803960274 w_k1_IBM_s117
                                 + 0.0159093234574187 w_k1_INTC_s000
                                 + 0.00898138767760497 w_k1_INTC_s001
                                 + 0.0161309732825511 w_k1_INTC_s002
                                 + 0.00659389532745714 w_k1_INTC_s003
                                 + 0.010912839717093 w_k1_INTC_s004
                                 + 0.0123901841129231 w_k1_INTC_s005
                                 + 0.0248194167928512 w_k1_INTC_s006
                                 + 0.013891217371814 w_k1_INTC_s007
                                 + 0.0256316952155363 w_k1_INTC_s010
                                 + 0.0140071651844143 w_k1_INTC_s011
                                 + 0.00548763927576982 w_k1_INTC_s012
                                 + 0.014423184189326 w_k1_INTC_s013
                                 + 0.0183977469881092 w_k1_INTC_s014
                                 + 0.0134631814640386 w_k1_INTC_s015
                                 + 0.0119170948828066 w_k1_INTC_s016
                                 + 0.00615656579707534 w_k1_INTC_s017
                                 + 0.05537571798362 w_k1_INTC_s100
                                 + 0.0266940049876646 w_k1_INTC_s101
                                 + 0.03686221165759 w_k1_INTC_s102
                                 + 0.0195248113375441 w_k1_INTC_s103
                                 + 0.0293573229069845 w_k1_INTC_s104
                                 + 0.0237796073606031 w_k1_INTC_s105
                                 + 0.0771472527608588 w_k1_INTC_s106
                                 + 0.028970211159744 w_k1_INTC_s107
                                 + 0.0465212917149139 w_k1_INTC_s110
                                 + 0.0963498823323856 w_k1_INTC_s111
                                 + 0.0721373207876905 w_k1_INTC_s112
                                 + 0.0774699623309756 w_k1_INTC_s113
                                 + 0.0363463457282081 w_k1_INTC_s114
                                 + 0.0808619467625556 w_k1_INTC_s115
                                 + 0.0748703648528248 w_k1_INTC_s116
                                 + 0.0256682658445274 w_k1_INTC_s117
                                 + 0.0151784549486497 w_k1_KO_s000
                                 + 0.0104942152462375 w_k1_KO_s001
                                 + 0.0156084085153929 w_k1_KO_s002
                                 + 0.00791601280182421 w_k1_KO_s003
                                 + 0.011073256199849 w_k1_KO_s004
                                 + 0.0137518207714064 w_k1_KO_s005
                                 + 0.0235807406131826 w_k1_KO_s006
                                 + 0.0143063180035893 w_k1_KO_s007
                                 + 0.0218225519184995 w_k1_KO_s010
                                 + 0.0130320415060647 w_k1_KO_s011
                                 + 0.00612613910141005 w_k1_KO_s012
                                 + 0.0151399748470816 w_k1_KO_s013
                                 + 0.0176490099502863 w_k1_KO_s014
                                 + 0.0129047614771387 w_k1_KO_s015
                                 + 0.012834016342774 w_k1_KO_s016
                                 + 0.00563621975506972 w_k1_KO_s017
                                 + 0.0539916597246714 w_k1_KO_s100
                                 + 0.0218936388653719 w_k1_KO_s101
                                 + 0.0334872621767569 w_k1_KO_s102
                                 + 0.0176124560718308 w_k1_KO_s103
                                 + 0.0334292522359262 w_k1_KO_s104
                                 + 0.0248694253709609 w_k1_KO_s105
                                 + 0.0725246039447939 w_k1_KO_s106
                                 + 0.030911875705379 w_k1_KO_s107
                                 + 0.0519898789105475 w_k1_KO_s110
                                 + 0.07388812089877 w_k1_KO_s111
                                 + 0.0810414661536699 w_k1_KO_s112
                                 + 0.0634895330988224 w_k1_KO_s113
                                 + 0.0430427737667135 w_k1_KO_s114
                                 + 0.081778752404835 w_k1_KO_s115
                                 + 0.0783488535990867 w_k1_KO_s116
                                 + 0.0322814380986374 w_k1_KO_s117
                                 + 0.0114903712321086 w_k1_MAT_s000
                                 + 0.0113617564983491 w_k1_MAT_s001
                                 + 0.0161876319498197 w_k1_MAT_s002
                                 + 0.0105619427044878 w_k1_MAT_s003
                                 + 0.0112337362733595 w_k1_MAT_s004
                                 + 0.0161552394047562 w_k1_MAT_s005
                                 + 0.0251369759862242 w_k1_MAT_s006
                                 + 0.020947255222581 w_k1_MAT_s007
                                 + 0.0221468325176955 w_k1_MAT_s010
                                 + 0.0133675403342962 w_k1_MAT_s011
                                 + 0.00670627791764157 w_k1_MAT_s012
                                 + 0.0151864249846398 w_k1_MAT_s013
                                 + 0.0154332848575798 w_k1_MAT_s014
                                 + 0.0154356766049696 w_k1_MAT_s015
                                 + 0.0163642443739044 w_k1_MAT_s016
                                 + 0.00484152353863044 w_k1_MAT_s017
                                 + 0.0597053523601692 w_k1_MAT_s100
                                 + 0.0215154998273893 w_k1_MAT_s101
                                 + 0.0264822923122582 w_k1_MAT_s102
                                 + 0.0186186700930902 w_k1_MAT_s103
                                 + 0.0356942974794166 w_k1_MAT_s104
                                 + 0.0262138075816791 w_k1_MAT_s105
                                 + 0.0672424191567148 w_k1_MAT_s106
                                 + 0.0370748588495519 w_k1_MAT_s107
                                 + 0.0404973366845349 w_k1_MAT_s110
                                 + 0.071636090030809 w_k1_MAT_s111
                                 + 0.0720565681921787 w_k1_MAT_s112
                                 + 0.0768209416572952 w_k1_MAT_s113
                                 + 0.0537203533073479 w_k1_MAT_s114
                                 + 0.0842858998443662 w_k1_MAT_s115
                                 + 0.0955296526554691 w_k1_MAT_s116
                                 + 0.0371176883529858 w_k1_MAT_s117
                                 + 0.0148325504906235 w_k1_MSFT_s000
                                 + 0.0103634424297389 w_k1_MSFT_s001
                                 + 0.015024386528439 w_k1_MSFT_s002
                                 + 0.00760504627555711 w_k1_MSFT_s003
                                 + 0.011806403094886 w_k1_MSFT_s004
                                 + 0.0129636488603927 w_k1_MSFT_s005
                                 + 0.0259442029528774 w_k1_MSFT_s006
                                 + 0.015421416461362 w_k1_MSFT_s007
                                 + 0.0252547611417162 w_k1_MSFT_s010
                                 + 0.0137043561013244 w_k1_MSFT_s011
                                 + 0.00624666765441374 w_k1_MSFT_s012
                                 + 0.0137961164780834 w_k1_MSFT_s013
                                 + 0.0164139929026072 w_k1_MSFT_s014
                                 + 0.0141850080806282 w_k1_MSFT_s015
                                 + 0.0119764367197712 w_k1_MSFT_s016
                                 + 0.00584079733123954 w_k1_MSFT_s017
                                 + 0.0558893040720524 w_k1_MSFT_s100
                                 + 0.0257344241501836 w_k1_MSFT_s101
                                 + 0.0325732124826775 w_k1_MSFT_s102
                                 + 0.0175981571826798 w_k1_MSFT_s103
                                 + 0.0352959361535783 w_k1_MSFT_s104
                                 + 0.0231949852804474 w_k1_MSFT_s105
                                 + 0.0810073242391142 w_k1_MSFT_s106
                                 + 0.032257686567262 w_k1_MSFT_s107
                                 + 0.0493879742108322 w_k1_MSFT_s110
                                 + 0.0868405474698223 w_k1_MSFT_s111
                                 + 0.0662904225041349 w_k1_MSFT_s112
                                 + 0.0806987858814249 w_k1_MSFT_s113
                                 + 0.0428299900657918 w_k1_MSFT_s114
                                 + 0.0787179337269674 w_k1_MSFT_s115
                                 + 0.0830491009718956 w_k1_MSFT_s116
                                 + 0.0310015139710276 w_k1_MSFT_s117
                                 + 0.0152657131891185 w_k1_PG_s000
                                 + 0.00893587366255117 w_k1_PG_s001
                                 + 0.0142359807435263 w_k1_PG_s002
                                 + 0.0076730228735982 w_k1_PG_s003
                                 + 0.0119031160438076 w_k1_PG_s004
                                 + 0.0139203473785621 w_k1_PG_s005
                                 + 0.0239506927587785 w_k1_PG_s006
                                 + 0.0145431921399364 w_k1_PG_s007
                                 + 0.0211603133246583 w_k1_PG_s010
                                 + 0.0132793852900037 w_k1_PG_s011
                                 + 0.00540650041394366 w_k1_PG_s012
                                 + 0.0146222603944876 w_k1_PG_s013
                                 + 0.0182825137328758 w_k1_PG_s014
                                 + 0.0145363706902206 w_k1_PG_s015
                                 + 0.0107513194009688 w_k1_PG_s016
                                 + 0.00578581674353551 w_k1_PG_s017
                                 + 0.0532131108684125 w_k1_PG_s100
                                 + 0.0243537076430461 w_k1_PG_s101
                                 + 0.0311787449847947 w_k1_PG_s102
                                 + 0.0188582451710938 w_k1_PG_s103
                                 + 0.0334568289831028 w_k1_PG_s104
                                 + 0.0229925555818985 w_k1_PG_s105
                                 + 0.0710998227651125 w_k1_PG_s106
                                 + 0.0314813047351 w_k1_PG_s107
                                 + 0.0483503241988757 w_k1_PG_s110
                                 + 0.0780883713727772 w_k1_PG_s111
                                 + 0.0705301023226141 w_k1_PG_s112
                                 + 0.0712278498701503 w_k1_PG_s113
                                 + 0.0445266185474279 w_k1_PG_s114
                                 + 0.080538106385395 w_k1_PG_s115
                                 + 0.0753805023194604 w_k1_PG_s116
                                 + 0.030893019696274 w_k1_PG_s117 >= 1.229
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
