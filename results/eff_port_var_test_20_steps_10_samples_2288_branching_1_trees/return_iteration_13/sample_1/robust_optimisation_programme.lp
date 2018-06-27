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
 w_k1_ADM_s0_return_constraint:  1.17856 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.935785 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.846649 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.981262 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.938121 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.0505 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.7709 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.942502 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.831164 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.08731 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0211 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.02382 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.26054 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.80138 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.35616 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.35056 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.00815 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26304 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.916089 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.25443 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.910789 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.44003 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.929639 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.742949 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.09241 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.07185 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.05856 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.910082 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.843252 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.642835 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.899116 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.11313 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.14419 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.50755 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.868357 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.31039 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.819349 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.05231 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24678 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.866259 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.791302 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.17812 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.853672 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.18816 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.693349 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.866571 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.645933 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.04867 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.18686 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.08665 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.26362 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.825946 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.51519 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.29783 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.814048 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.996545 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.670973 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.955267 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.697626 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.12997 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.01139 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.733782 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.982825 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.39184 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.32001 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.23923 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.750644 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.734164 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.768089 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.851438 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.12525 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.75673 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.829803 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.46703 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.760803 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.20185 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29512 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.850944 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.768799 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.1898 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.876152 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.23146 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.5246 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.857103 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.707579 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.20077 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.52043 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.09376 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.04543 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.747013 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.76269 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.28697 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.875645 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.20114 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.611664 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.838326 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.634932 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.24317 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.841036 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.628765 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.927084 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.49334 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.1341 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.960554 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.894861 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.591678 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.875275 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.977032 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.13131 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 2.00538 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.718651 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.38881 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.629704 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.20756 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19301 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.916448 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.786383 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.13423 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.894735 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0882 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.718427 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.07886 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.89784 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.1673 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.45592 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.01883 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.31642 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.863656 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.21068 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.14913 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.834386 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.04274 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.674247 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.00371 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.771958 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.28643 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.963417 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.851992 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.07768 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.37406 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.04429 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.939826 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.864874 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.650047 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.777366 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.00423 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.09156 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.61134 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.734177 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.22545 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.771472 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.941408 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.16976 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.948201 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.856659 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.10725 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.764284 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.23761 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.763558 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.629969 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.61387 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.07218 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.799272 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.62458 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.645972 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.00249 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.71596 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.12635 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.211 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.767296 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.842458 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.706513 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.566092 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.599006 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.07297 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.646076 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.861708 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.07067 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.21278 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.63145 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.889951 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.08978 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.03729 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.741132 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.853796 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.939684 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.2177 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.22256 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.779106 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.89927 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15801 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.885728 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.769909 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.01609 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.860654 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.0479 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.9261 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.19338 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0725 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.16824 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.40382 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.11155 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.13956 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.911182 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.24635 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.19992 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01343 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.05542 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.878324 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.98906 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.8922 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.301 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.03427 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.00068 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.03889 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.42629 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00986 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.21146 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.927337 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.840013 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.996207 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.14274 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.17699 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.40369 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.989758 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.22939 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.943325 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.13389 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.27823 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.911554 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.786702 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.02455 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.868926 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.21519 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.709133 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.829572 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.640386 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.08567 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.0523 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.1782 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.833327 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.887596 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.51136 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.26905 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.987506 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.932934 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.876429 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.783971 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.665518 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.963243 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.964319 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.61506 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.918517 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.30048 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.11322 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.19645 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.88133 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.716135 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.938867 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.956889 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.915645 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.32183 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.921701 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.24587 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.782822 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.40548 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.20712 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.859512 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.854582 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.17664 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.752942 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.33745 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.623102 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.603662 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.385547 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.08141 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.916515 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.50399 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.810835 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.805004 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.9701 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.34775 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.16963 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.837273 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.797733 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.622127 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.449966 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.568656 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.11134 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.528331 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.925153 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.983053 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.56489 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.73185 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.900964 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.09716 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.789026 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.74588 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.16108 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.2139 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.12974 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.34674 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.708576 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.93502 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.21574 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.914571 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.800147 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.991828 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.927547 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.07241 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.696361 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.09894 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.927755 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.1968 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.41304 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.986376 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.12257 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.82945 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.40465 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.24594 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.00908 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0858 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.799006 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.950564 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.88154 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.36652 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.892354 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.842546 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.06266 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.25906 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.00711 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.05803 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.938161 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.704482 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.92519 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.15307 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.08598 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.62195 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.859104 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.08801 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.78838 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.04634 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12524 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.94548 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.861367 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.05034 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.922085 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11427 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.762413 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.956965 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.869032 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.10916 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.12699 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.1859 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.11818 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.928482 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.22075 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.25262 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00079 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.987787 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.80442 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.913505 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.815195 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.971873 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.958026 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.779209 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.963807 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.15626 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.988855 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.07962 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.925964 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.79682 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.917091 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.993002 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.924302 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.33883 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.824628 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.143 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.757934 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08497 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.135999543171559 z_k1_s0000
                                 - 0.0589999225228629 z_k1_s0001
                                 - 0.167999486506005 z_k1_s0002
                                 - 0.0989996356910861 z_k1_s0003
                                 - 0.0519999727184238 z_k1_s0004
                                 - 0.0900002186278363 z_k1_s0005
                                 - 0.0719999157024688 z_k1_s0006
                                 - 0.18800020708945 z_k1_s0007
                                 - 0.102999950341543 z_k1_s0010
                                 - 0.133999617046272 z_k1_s0011
                                 - 0.107000002532478 z_k1_s0012
                                 - 0.157999711591877 z_k1_s0013
                                 - 0.283999661456289 z_k1_s0014
                                 - 0.0590000320412737 z_k1_s0015
                                 - 0.175000233978357 z_k1_s0016
                                 - 0.0730000507595423 z_k1_s0017
                                 - 0.0849999735333015 z_k1_s0020
                                 - 0.136000139053315 z_k1_s0021
                                 - 0.0740000569878296 z_k1_s0022
                                 - 0.130999847168951 z_k1_s0023
                                 - 0.0569999715811533 z_k1_s0024
                                 - 0.11300042898553 z_k1_s0025
                                 - 0.235999627739467 z_k1_s0026
                                 - 0.0750000246646863 z_k1_s0027
                                 - 0.627999953507007 z_k1_s0030
                                 - 0.422000293675286 z_k1_s0031
                                 - 0.292000711529176 z_k1_s0032
                                 - 0.178000071800445 z_k1_s0033
                                 - 0.19200012741624 z_k1_s0034
                                 - 0.285999603921756 z_k1_s0035
                                 - 0.273000692107774 z_k1_s0036
                                 - 0.345000903388962 z_k1_s0037
                                 - 0.0659999616938362 z_k1_s0040
                                 - 0.0499999596529123 z_k1_s0041
                                 - 0.0529999522480897 z_k1_s0042
                                 - 0.112999679551704 z_k1_s0043
                                 - 0.129000179992362 z_k1_s0044
                                 - 0.116999960411785 z_k1_s0045
                                 - 0.0609999532685516 z_k1_s0046
                                 - 0.0339999576119883 z_k1_s0047
                                 - 0.389001709216559 z_k1_s0050
                                 - 0.547001668486185 z_k1_s0051
                                 - 0.249000319424097 z_k1_s0052
                                 - 0.202000306207153 z_k1_s0053
                                 - 0.290000142381212 z_k1_s0054
                                 - 0.437999308050571 z_k1_s0055
                                 - 0.458000895251105 z_k1_s0056
                                 - 0.286000396741768 z_k1_s0057
                                 - 0.122000132814206 z_k1_s0060
                                 - 0.139000019353801 z_k1_s0061
                                 - 0.0309999960429284 z_k1_s0062
                                 - 0.0649997691221182 z_k1_s0063
                                 - 0.0590002150495956 z_k1_s0064
                                 - 0.0460000312250158 z_k1_s0065
                                 - 0.0520000452978507 z_k1_s0066
                                 - 0.0610000664071031 z_k1_s0067
                                 - 0.283000770476915 z_k1_s0070
                                 - 0.137000172586913 z_k1_s0071
                                 - 0.391000440510058 z_k1_s0072
                                 - 0.163000137172702 z_k1_s0073
                                 - 0.25700097569125 z_k1_s0074
                                 - 0.0790000164799848 z_k1_s0075
                                 - 0.223000199124624 z_k1_s0076
                                 - 0.165000043017696 z_k1_s0077
                                 - 0.927004543197586 z_k1_s0100
                                 - 0.352001316841326 z_k1_s0101
                                 - 0.415000802586037 z_k1_s0102
                                 - 0.483003245533013 z_k1_s0103
                                 - 0.826001024056111 z_k1_s0104
                                 - 0.232000614417316 z_k1_s0105
                                 - 0.234000904975297 z_k1_s0106
                                 - 0.249000853352841 z_k1_s0107
                                 - 0.569000320884353 z_k1_s0110
                                 - 0.316999155406599 z_k1_s0111
                                 - 0.450997446163928 z_k1_s0112
                                 - 0.341999002769819 z_k1_s0113
                                 - 0.717996138646218 z_k1_s0114
                                 - 0.342999196395794 z_k1_s0115
                                 - 0.389000037113007 z_k1_s0116
                                 - 0.711996513286108 z_k1_s0117
                                 - 1.3569970652876 z_k1_s0120
                                 - 0.368000334851509 z_k1_s0121
                                 - 0.585000113790343 z_k1_s0122
                                 - 1.42300040631607 z_k1_s0123
                                 - 1.13100109850182 z_k1_s0124
                                 - 1.73600152591252 z_k1_s0125
                                 - 1.00100400157228 z_k1_s0126
                                 - 0.635000074240773 z_k1_s0127
                                 - 0.165999956784513 z_k1_s0130
                                 - 0.277000914735883 z_k1_s0131
                                 - 0.322000399526723 z_k1_s0132
                                 - 0.299999749273167 z_k1_s0133
                                 - 0.150000113436651 z_k1_s0134
                                 - 0.361999410896207 z_k1_s0135
                                 - 0.351000279769768 z_k1_s0136
                                 - 0.460000570752461 z_k1_s0137
                                 - 0.376999856743039 z_k1_s0140
                                 - 0.108000046902552 z_k1_s0141
                                 - 0.116000086228681 z_k1_s0142
                                 - 0.269999512256127 z_k1_s0143
                                 - 0.506000430376857 z_k1_s0144
                                 - 0.357000968428217 z_k1_s0145
                                 - 0.33500073928131 z_k1_s0146
                                 - 0.350999365932966 z_k1_s0147
                                 - 0.217000044260445 z_k1_s0150
                                 - 0.337999481185479 z_k1_s0151
                                 - 0.276998923995294 z_k1_s0152
                                 - 0.29899894281801 z_k1_s0153
                                 - 0.273999224428516 z_k1_s0154
                                 - 0.351000401307861 z_k1_s0155
                                 - 0.31499956250717 z_k1_s0156
                                 - 0.150999987792297 z_k1_s0157
                                 - 0.19299927331722 z_k1_s0160
                                 - 0.695997519460595 z_k1_s0161
                                 - 0.402997280358617 z_k1_s0162
                                 - 0.351998442394874 z_k1_s0163
                                 - 0.512997617604268 z_k1_s0164
                                 - 0.222999365293306 z_k1_s0165
                                 - 0.652998046826006 z_k1_s0166
                                 - 0.262999158329187 z_k1_s0167
                                 - 0.362000659554458 z_k1_s0170
                                 - 0.129000051827852 z_k1_s0171
                                 - 0.224000468519924 z_k1_s0172
                                 - 0.164000147809104 z_k1_s0173
                                 - 0.304999352778532 z_k1_s0174
                                 - 0.25299943922939 z_k1_s0175
                                 - 0.0910000309111164 z_k1_s0176
                                 - 0.29399993168196 z_k1_s0177
                                 - 0.269999951260066 z_k1_s1000
                                 - 0.272999941081847 z_k1_s1001
                                 - 0.722998414349046 z_k1_s1002
                                 - 0.338001166053353 z_k1_s1003
                                 - 0.3929995339531 z_k1_s1004
                                 - 0.247000029293076 z_k1_s1005
                                 - 0.277999924118149 z_k1_s1006
                                 - 0.368998458979188 z_k1_s1007
                                 - 0.55499922188684 z_k1_s1010
                                 - 0.263999141271833 z_k1_s1011
                                 - 0.317999065563526 z_k1_s1012
                                 - 0.949997039432379 z_k1_s1013
                                 - 0.778997279175353 z_k1_s1014
                                 - 0.403998782124815 z_k1_s1015
                                 - 0.557999217680823 z_k1_s1016
                                 - 1.05799688801576 z_k1_s1017
                                 - 0.306999191350866 z_k1_s1020
                                 - 0.124999879800992 z_k1_s1021
                                 - 0.10799991681284 z_k1_s1022
                                 - 0.145999869145151 z_k1_s1023
                                 - 0.287000053070003 z_k1_s1024
                                 - 0.339999259763044 z_k1_s1025
                                 - 0.508999429480107 z_k1_s1026
                                 - 0.164999944661223 z_k1_s1027
                                 - 1.68499393852442 z_k1_s1030
                                 - 0.599999208600209 z_k1_s1031
                                 - 0.468999656635436 z_k1_s1032
                                 - 2.01499393785391 z_k1_s1033
                                 - 0.889998681223851 z_k1_s1034
                                 - 1.45199652025475 z_k1_s1035
                                 - 0.886000366950776 z_k1_s1036
                                 - 0.694999155728472 z_k1_s1037
                                 - 0.196999958999047 z_k1_s1040
                                 - 0.28800060528554 z_k1_s1041
                                 - 0.195000008328978 z_k1_s1042
                                 - 0.351000737691752 z_k1_s1043
                                 - 0.405000851182791 z_k1_s1044
                                 - 0.444999864584166 z_k1_s1045
                                 - 0.197999934334081 z_k1_s1046
                                 - 0.329999087557256 z_k1_s1047
                                 - 0.207999845778731 z_k1_s1050
                                 - 0.319999840344193 z_k1_s1051
                                 - 0.220999857158649 z_k1_s1052
                                 - 0.289000046030516 z_k1_s1053
                                 - 0.452999413515725 z_k1_s1054
                                 - 1.01099840156162 z_k1_s1055
                                 - 0.29899992543816 z_k1_s1056
                                 - 0.562000957608729 z_k1_s1057
                                 - 0.358999286937244 z_k1_s1060
                                 - 0.290001030228246 z_k1_s1061
                                 - 0.400000576866342 z_k1_s1062
                                 - 0.276001186837967 z_k1_s1063
                                 - 0.290001030228246 z_k1_s1064
                                 - 0.332000560399084 z_k1_s1065
                                 - 0.610000947721188 z_k1_s1066
                                 - 0.163000119473006 z_k1_s1067
                                 - 0.607999194007476 z_k1_s1070
                                 - 0.259999906487452 z_k1_s1071
                                 - 0.252999948243951 z_k1_s1072
                                 - 0.535998683217009 z_k1_s1073
                                 - 0.327000070989637 z_k1_s1074
                                 - 0.412000363189438 z_k1_s1075
                                 - 0.374999643616478 z_k1_s1076
                                 - 0.520001129375138 z_k1_s1077
                                 - 0.128000030185121 z_k1_s1100
                                 - 0.123000046831013 z_k1_s1101
                                 - 0.133999924650058 z_k1_s1102
                                 - 0.221000034291499 z_k1_s1103
                                 - 0.12099993940938 z_k1_s1104
                                 - 0.323000379195239 z_k1_s1105
                                 - 0.112000026411981 z_k1_s1106
                                 - 0.26399963445685 z_k1_s1107
                                 - 0.471999751269823 z_k1_s1110
                                 - 0.368000246929515 z_k1_s1111
                                 - 0.361000456591425 z_k1_s1112
                                 - 0.78499902622566 z_k1_s1113
                                 - 0.550998968858636 z_k1_s1114
                                 - 0.506000925828748 z_k1_s1115
                                 - 0.270000197392869 z_k1_s1116
                                 - 0.951000998479671 z_k1_s1117
                                 - 0.397999578266549 z_k1_s1120
                                 - 0.502999362804396 z_k1_s1121
                                 - 0.997995940201494 z_k1_s1122
                                 - 0.315999215790341 z_k1_s1123
                                 - 1.15399517574425 z_k1_s1124
                                 - 1.45599502506177 z_k1_s1125
                                 - 0.921998638778987 z_k1_s1126
                                 - 0.732996823414879 z_k1_s1127
                                 - 0.136999942255243 z_k1_s1130
                                 - 0.268999334582976 z_k1_s1131
                                 - 0.250000024378169 z_k1_s1132
                                 - 0.0859999566740951 z_k1_s1133
                                 - 0.188999262141967 z_k1_s1134
                                 - 0.3929998508665 z_k1_s1135
                                 - 0.203000284819048 z_k1_s1136
                                 - 0.0890000603906234 z_k1_s1137
                                 - 0.291000318938539 z_k1_s1140
                                 - 0.940002554494641 z_k1_s1141
                                 - 0.571002770369452 z_k1_s1142
                                 - 0.224000264061998 z_k1_s1143
                                 - 0.622000106480074 z_k1_s1144
                                 - 0.473999064184143 z_k1_s1145
                                 - 0.480999530455567 z_k1_s1146
                                 - 0.254000444065177 z_k1_s1147
                                 - 1.3570034758122 z_k1_s1150
                                 - 0.580998053421406 z_k1_s1151
                                 - 0.278000142880708 z_k1_s1152
                                 - 0.409000164389204 z_k1_s1153
                                 - 0.625997972412797 z_k1_s1154
                                 - 0.816002777047576 z_k1_s1155
                                 - 0.393000064525487 z_k1_s1156
                                 - 1.32999889241335 z_k1_s1157
                                 - 0.887998368865339 z_k1_s1160
                                 - 0.380999957887127 z_k1_s1161
                                 - 0.313000331488547 z_k1_s1162
                                 - 0.392999948933962 z_k1_s1163
                                 - 0.338999989223205 z_k1_s1164
                                 - 0.961999443688021 z_k1_s1165
                                 - 0.761997978573343 z_k1_s1166
                                 - 0.804998995808329 z_k1_s1167
                                 - 0.369000264486326 z_k1_s1170
                                 - 0.202000122279222 z_k1_s1171
                                 - 0.230000198616972 z_k1_s1172
                                 - 0.600999249140954 z_k1_s1173
                                 - 0.287999137330129 z_k1_s1174
                                 - 0.190000023648716 z_k1_s1175
                                 - 0.231000053036086 z_k1_s1176
                                 - 0.196000072963969 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.863537 w_k1_ADM_s000
                                 + 0.650622 w_k1_AXP_s000
                                 + 0.796493 w_k1_DIS_s000
                                 + 0.89491 w_k1_FDX_s000
                                 + 0.706116 w_k1_IBM_s000
                                 + 0.684623 w_k1_INTC_s000
                                 + 0.813046 w_k1_KO_s000
                                 + 0.947152 w_k1_MAT_s000
                                 + 0.974195 w_k1_MSFT_s000
                                 + 0.871977 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.915037 w_k1_ADM_s000
                                 + 0.693175 w_k1_AXP_s000
                                 + 0.992158 w_k1_DIS_s000
                                 + 0.968839 w_k1_FDX_s000
                                 + 0.930456 w_k1_IBM_s000
                                 + 0.797048 w_k1_INTC_s000
                                 + 0.90066 w_k1_KO_s000
                                 + 0.931775 w_k1_MAT_s000
                                 + 1.04286 w_k1_MSFT_s000
                                 + 0.862706 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09247 w_k1_ADM_s000
                                 + 1.19895 w_k1_AXP_s000
                                 + 1.35337 w_k1_DIS_s000
                                 + 1.18171 w_k1_FDX_s000
                                 + 0.897256 w_k1_IBM_s000
                                 + 0.834883 w_k1_INTC_s000
                                 + 1.16408 w_k1_KO_s000 + 1.16828 w_k1_MAT_s000
                                 + 1.11031 w_k1_MSFT_s000
                                 + 1.20449 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.681177 w_k1_ADM_s000
                                 + 0.841414 w_k1_AXP_s000
                                 + 0.662105 w_k1_DIS_s000
                                 + 0.745148 w_k1_FDX_s000
                                 + 0.866264 w_k1_IBM_s000
                                 + 0.777107 w_k1_INTC_s000
                                 + 0.90548 w_k1_KO_s000 + 1.18601 w_k1_MAT_s000
                                 + 0.87249 w_k1_MSFT_s000
                                 + 0.936501 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.41051 w_k1_ADM_s000
                                 + 1.68758 w_k1_AXP_s000 + 1.8451 w_k1_DIS_s000
                                 + 1.58464 w_k1_FDX_s000
                                 + 1.42789 w_k1_IBM_s000
                                 + 1.06164 w_k1_INTC_s000
                                 + 1.21647 w_k1_KO_s000 + 1.53645 w_k1_MAT_s000
                                 + 1.18254 w_k1_MSFT_s000
                                 + 1.21223 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.11086 w_k1_ADM_s000
                                 + 0.76002 w_k1_AXP_s000
                                 + 1.09486 w_k1_DIS_s000 + 1.0357 w_k1_FDX_s000
                                 + 0.569436 w_k1_IBM_s000
                                 + 0.733852 w_k1_INTC_s000
                                 + 0.778941 w_k1_KO_s000
                                 + 0.72436 w_k1_MAT_s000
                                 + 1.01418 w_k1_MSFT_s000
                                 + 0.982235 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.829103 w_k1_ADM_s000
                                 + 0.737565 w_k1_AXP_s000
                                 + 0.735117 w_k1_DIS_s000
                                 + 0.721265 w_k1_FDX_s000
                                 + 1.1469 w_k1_IBM_s000
                                 + 0.871989 w_k1_INTC_s000
                                 + 0.887491 w_k1_KO_s000
                                 + 0.900549 w_k1_MAT_s000
                                 + 0.747714 w_k1_MSFT_s000
                                 + 0.960542 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.979645 w_k1_ADM_s000
                                 + 1.05294 w_k1_AXP_s000
                                 + 1.15589 w_k1_DIS_s000
                                 + 0.888323 w_k1_FDX_s000
                                 + 1.44249 w_k1_IBM_s000
                                 + 0.905213 w_k1_INTC_s000
                                 + 1.1159 w_k1_KO_s000 + 1.38256 w_k1_MAT_s000
                                 + 0.992997 w_k1_MSFT_s000
                                 + 1.12395 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.929062 w_k1_ADM_s001
                                 + 1.06588 w_k1_AXP_s001
                                 + 1.00771 w_k1_DIS_s001
                                 + 0.970887 w_k1_FDX_s001
                                 + 0.895293 w_k1_IBM_s001
                                 + 0.974151 w_k1_INTC_s001
                                 + 0.919649 w_k1_KO_s001
                                 + 1.06453 w_k1_MAT_s001
                                 + 0.846875 w_k1_MSFT_s001
                                 + 1.01091 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.80677 w_k1_ADM_s001
                                 + 0.853603 w_k1_AXP_s001
                                 + 0.575179 w_k1_DIS_s001
                                 + 0.806198 w_k1_FDX_s001
                                 + 0.611606 w_k1_IBM_s001
                                 + 0.766555 w_k1_INTC_s001
                                 + 0.580098 w_k1_KO_s001
                                 + 0.781739 w_k1_MAT_s001
                                 + 0.706118 w_k1_MSFT_s001
                                 + 0.81235 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.27575 w_k1_ADM_s001
                                 + 1.53672 w_k1_AXP_s001
                                 + 1.39103 w_k1_DIS_s001 + 1.2313 w_k1_FDX_s001
                                 + 1.3641 w_k1_IBM_s001
                                 + 0.98532 w_k1_INTC_s001
                                 + 0.935722 w_k1_KO_s001
                                 + 1.26925 w_k1_MAT_s001
                                 + 0.973722 w_k1_MSFT_s001
                                 + 1.18074 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.92109 w_k1_ADM_s001
                                 + 0.766106 w_k1_AXP_s001
                                 + 0.763957 w_k1_DIS_s001
                                 + 0.802886 w_k1_FDX_s001
                                 + 0.8709 w_k1_IBM_s001
                                 + 0.71196 w_k1_INTC_s001
                                 + 0.730881 w_k1_KO_s001
                                 + 0.606256 w_k1_MAT_s001
                                 + 0.696639 w_k1_MSFT_s001
                                 + 0.851067 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.30397 w_k1_ADM_s001
                                 + 0.933878 w_k1_AXP_s001
                                 + 0.983415 w_k1_DIS_s001
                                 + 1.05277 w_k1_FDX_s001
                                 + 0.803985 w_k1_IBM_s001
                                 + 0.813192 w_k1_INTC_s001
                                 + 0.958014 w_k1_KO_s001
                                 + 0.826955 w_k1_MAT_s001
                                 + 0.91782 w_k1_MSFT_s001
                                 + 0.965538 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.882875 w_k1_ADM_s001
                                 + 1.05809 w_k1_AXP_s001
                                 + 1.53196 w_k1_DIS_s001
                                 + 1.07488 w_k1_FDX_s001
                                 + 1.59181 w_k1_IBM_s001
                                 + 1.09773 w_k1_INTC_s001
                                 + 1.40828 w_k1_KO_s001 + 1.04367 w_k1_MAT_s001
                                 + 1.09288 w_k1_MSFT_s001 + 1.1116 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.906948 w_k1_ADM_s001
                                 + 1.06413 w_k1_AXP_s001
                                 + 0.830095 w_k1_DIS_s001
                                 + 0.895469 w_k1_FDX_s001
                                 + 1.26612 w_k1_IBM_s001
                                 + 0.861206 w_k1_INTC_s001
                                 + 0.876189 w_k1_KO_s001
                                 + 1.33916 w_k1_MAT_s001
                                 + 0.875755 w_k1_MSFT_s001
                                 + 0.967653 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.31008 w_k1_ADM_s001
                                 + 1.34427 w_k1_AXP_s001 + 1.5081 w_k1_DIS_s001
                                 + 1.37254 w_k1_FDX_s001
                                 + 0.803412 w_k1_IBM_s001
                                 + 1.05579 w_k1_INTC_s001
                                 + 1.12553 w_k1_KO_s001
                                 + 0.904483 w_k1_MAT_s001
                                 + 1.10452 w_k1_MSFT_s001 + 1.1795 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07178 w_k1_ADM_s002
                                 + 1.77378 w_k1_AXP_s002 + 1.5357 w_k1_DIS_s002
                                 + 1.28795 w_k1_FDX_s002
                                 + 1.74459 w_k1_IBM_s002
                                 + 1.21618 w_k1_INTC_s002
                                 + 1.57539 w_k1_KO_s002 + 3.13713 w_k1_MAT_s002
                                 + 1.17051 w_k1_MSFT_s002
                                 + 1.10261 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.925832 w_k1_ADM_s002
                                 + 1.31391 w_k1_AXP_s002
                                 + 1.42961 w_k1_DIS_s002
                                 + 1.19452 w_k1_FDX_s002
                                 + 1.18738 w_k1_IBM_s002
                                 + 1.15035 w_k1_INTC_s002
                                 + 1.37431 w_k1_KO_s002 + 1.61711 w_k1_MAT_s002
                                 + 1.14683 w_k1_MSFT_s002
                                 + 1.15813 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.885901 w_k1_ADM_s002
                                 + 0.930743 w_k1_AXP_s002
                                 + 1.27258 w_k1_DIS_s002
                                 + 0.868138 w_k1_FDX_s002
                                 + 1.19626 w_k1_IBM_s002
                                 + 0.832701 w_k1_INTC_s002
                                 + 1.28824 w_k1_KO_s002 + 2.23136 w_k1_MAT_s002
                                 + 1.0993 w_k1_MSFT_s002 + 1.0424 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09482 w_k1_ADM_s002
                                 + 1.29373 w_k1_AXP_s002
                                 + 1.14773 w_k1_DIS_s002
                                 + 1.12087 w_k1_FDX_s002 + 0.892 w_k1_IBM_s002
                                 + 0.889657 w_k1_INTC_s002
                                 + 1.11542 w_k1_KO_s002 + 1.35749 w_k1_MAT_s002
                                 + 1.00829 w_k1_MSFT_s002
                                 + 1.03754 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.93116 w_k1_ADM_s002
                                 + 1.24625 w_k1_AXP_s002
                                 + 1.16915 w_k1_DIS_s002
                                 + 0.998656 w_k1_FDX_s002
                                 + 1.15408 w_k1_IBM_s002
                                 + 0.783628 w_k1_INTC_s002
                                 + 1.17311 w_k1_KO_s002 + 1.81038 w_k1_MAT_s002
                                 + 0.914296 w_k1_MSFT_s002
                                 + 1.02903 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.839269 w_k1_ADM_s002
                                 + 0.875779 w_k1_AXP_s002
                                 + 0.856159 w_k1_DIS_s002
                                 + 0.864432 w_k1_FDX_s002
                                 + 1.00666 w_k1_IBM_s002
                                 + 0.889955 w_k1_INTC_s002
                                 + 1.14546 w_k1_KO_s002 + 1.25001 w_k1_MAT_s002
                                 + 0.974203 w_k1_MSFT_s002
                                 + 0.872423 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.982835 w_k1_ADM_s002
                                 + 0.942077 w_k1_AXP_s002
                                 + 0.822824 w_k1_DIS_s002
                                 + 0.87866 w_k1_FDX_s002
                                 + 0.728706 w_k1_IBM_s002
                                 + 0.79191 w_k1_INTC_s002
                                 + 0.904189 w_k1_KO_s002
                                 + 1.07577 w_k1_MAT_s002
                                 + 0.768372 w_k1_MSFT_s002
                                 + 0.867409 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.977895 w_k1_ADM_s002
                                 + 1.01002 w_k1_AXP_s002
                                 + 1.08737 w_k1_DIS_s002
                                 + 0.977387 w_k1_FDX_s002
                                 + 1.53403 w_k1_IBM_s002
                                 + 0.836404 w_k1_INTC_s002
                                 + 1.37399 w_k1_KO_s002 + 1.79691 w_k1_MAT_s002
                                 + 0.908555 w_k1_MSFT_s002
                                 + 1.04565 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.06238 w_k1_ADM_s003
                                 + 1.22573 w_k1_AXP_s003
                                 + 1.08549 w_k1_DIS_s003 + 1.0393 w_k1_FDX_s003
                                 + 1.18059 w_k1_IBM_s003
                                 + 1.00918 w_k1_INTC_s003
                                 + 1.09043 w_k1_KO_s003 + 1.08525 w_k1_MAT_s003
                                 + 0.829456 w_k1_MSFT_s003
                                 + 1.08508 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.07978 w_k1_ADM_s003
                                 + 1.57669 w_k1_AXP_s003
                                 + 1.67381 w_k1_DIS_s003
                                 + 1.26868 w_k1_FDX_s003
                                 + 1.37515 w_k1_IBM_s003
                                 + 1.4074 w_k1_INTC_s003 + 1.35894 w_k1_KO_s003
                                 + 1.48701 w_k1_MAT_s003
                                 + 1.20168 w_k1_MSFT_s003
                                 + 1.20192 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.802393 w_k1_ADM_s003
                                 + 0.685447 w_k1_AXP_s003
                                 + 0.659057 w_k1_DIS_s003
                                 + 0.821544 w_k1_FDX_s003
                                 + 0.743224 w_k1_IBM_s003
                                 + 0.782866 w_k1_INTC_s003
                                 + 0.764458 w_k1_KO_s003
                                 + 0.825571 w_k1_MAT_s003
                                 + 0.832306 w_k1_MSFT_s003
                                 + 0.7811 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.809238 w_k1_ADM_s003
                                 + 0.605483 w_k1_AXP_s003
                                 + 0.913481 w_k1_DIS_s003
                                 + 0.759193 w_k1_FDX_s003
                                 + 1.0153 w_k1_IBM_s003
                                 + 1.07546 w_k1_INTC_s003
                                 + 0.971775 w_k1_KO_s003
                                 + 0.781779 w_k1_MAT_s003
                                 + 0.89128 w_k1_MSFT_s003
                                 + 0.966148 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0759 w_k1_ADM_s003
                                 + 1.13932 w_k1_AXP_s003
                                 + 1.21521 w_k1_DIS_s003
                                 + 1.23619 w_k1_FDX_s003
                                 + 0.807706 w_k1_IBM_s003
                                 + 1.00968 w_k1_INTC_s003
                                 + 1.05257 w_k1_KO_s003 + 1.0241 w_k1_MAT_s003
                                 + 1.02922 w_k1_MSFT_s003
                                 + 1.03301 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.875563 w_k1_ADM_s003
                                 + 0.815315 w_k1_AXP_s003
                                 + 0.797344 w_k1_DIS_s003
                                 + 0.909718 w_k1_FDX_s003
                                 + 1.07184 w_k1_IBM_s003
                                 + 0.864899 w_k1_INTC_s003
                                 + 0.91858 w_k1_KO_s003
                                 + 0.855945 w_k1_MAT_s003
                                 + 0.857634 w_k1_MSFT_s003
                                 + 1.01879 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.895522 w_k1_ADM_s003
                                 + 0.906971 w_k1_AXP_s003
                                 + 0.796247 w_k1_DIS_s003
                                 + 0.948041 w_k1_FDX_s003
                                 + 0.646028 w_k1_IBM_s003
                                 + 0.892584 w_k1_INTC_s003
                                 + 0.866624 w_k1_KO_s003
                                 + 0.583813 w_k1_MAT_s003
                                 + 0.753605 w_k1_MSFT_s003
                                 + 0.83313 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.955805 w_k1_ADM_s003
                                 + 0.960909 w_k1_AXP_s003
                                 + 0.99566 w_k1_DIS_s003
                                 + 0.985486 w_k1_FDX_s003
                                 + 0.900369 w_k1_IBM_s003
                                 + 1.12729 w_k1_INTC_s003
                                 + 0.935082 w_k1_KO_s003
                                 + 0.689645 w_k1_MAT_s003
                                 + 0.960241 w_k1_MSFT_s003
                                 + 0.873304 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.17839 w_k1_ADM_s004
                                 + 1.0172 w_k1_AXP_s004
                                 + 0.951634 w_k1_DIS_s004
                                 + 0.975384 w_k1_FDX_s004
                                 + 1.60936 w_k1_IBM_s004
                                 + 0.97237 w_k1_INTC_s004
                                 + 1.14878 w_k1_KO_s004 + 1.45628 w_k1_MAT_s004
                                 + 1.02303 w_k1_MSFT_s004 + 1.0812 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.928177 w_k1_ADM_s004
                                 + 0.83115 w_k1_AXP_s004 + 0.8254 w_k1_DIS_s004
                                 + 1.01628 w_k1_FDX_s004
                                 + 0.643257 w_k1_IBM_s004
                                 + 0.929172 w_k1_INTC_s004
                                 + 0.86474 w_k1_KO_s004
                                 + 0.700221 w_k1_MAT_s004
                                 + 0.996713 w_k1_MSFT_s004
                                 + 0.975094 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.56852 w_k1_ADM_s004
                                 + 1.04809 w_k1_AXP_s004
                                 + 1.03336 w_k1_DIS_s004
                                 + 1.17045 w_k1_FDX_s004
                                 + 0.832535 w_k1_IBM_s004
                                 + 0.86995 w_k1_INTC_s004
                                 + 0.913374 w_k1_KO_s004
                                 + 0.951399 w_k1_MAT_s004
                                 + 0.975032 w_k1_MSFT_s004
                                 + 1.05707 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.23322 w_k1_ADM_s004
                                 + 1.2361 w_k1_AXP_s004 + 1.36226 w_k1_DIS_s004
                                 + 1.23793 w_k1_FDX_s004
                                 + 1.37922 w_k1_IBM_s004
                                 + 1.17288 w_k1_INTC_s004
                                 + 1.25804 w_k1_KO_s004 + 1.17074 w_k1_MAT_s004
                                 + 1.21443 w_k1_MSFT_s004 + 1.1262 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.06564 w_k1_ADM_s004
                                 + 0.672725 w_k1_AXP_s004
                                 + 0.727821 w_k1_DIS_s004
                                 + 0.81162 w_k1_FDX_s004
                                 + 0.762814 w_k1_IBM_s004
                                 + 0.708526 w_k1_INTC_s004
                                 + 0.776589 w_k1_KO_s004
                                 + 0.610171 w_k1_MAT_s004
                                 + 0.777405 w_k1_MSFT_s004
                                 + 0.86915 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.11776 w_k1_ADM_s004
                                 + 0.820176 w_k1_AXP_s004 + 1.117 w_k1_DIS_s004
                                 + 0.904883 w_k1_FDX_s004
                                 + 1.0325 w_k1_IBM_s004
                                 + 1.07965 w_k1_INTC_s004
                                 + 1.07286 w_k1_KO_s004
                                 + 0.726004 w_k1_MAT_s004
                                 + 0.99993 w_k1_MSFT_s004
                                 + 1.01563 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.33276 w_k1_ADM_s004
                                 + 0.954972 w_k1_AXP_s004
                                 + 0.815656 w_k1_DIS_s004
                                 + 0.897105 w_k1_FDX_s004
                                 + 0.896971 w_k1_IBM_s004
                                 + 0.761594 w_k1_INTC_s004
                                 + 0.859912 w_k1_KO_s004
                                 + 0.785378 w_k1_MAT_s004
                                 + 0.858973 w_k1_MSFT_s004
                                 + 1.01688 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 2.01077 w_k1_ADM_s004
                                 + 1.5294 w_k1_AXP_s004 + 1.90818 w_k1_DIS_s004
                                 + 1.52349 w_k1_FDX_s004
                                 + 1.84318 w_k1_IBM_s004
                                 + 1.07237 w_k1_INTC_s004
                                 + 1.65356 w_k1_KO_s004 + 1.43443 w_k1_MAT_s004
                                 + 1.331 w_k1_MSFT_s004 + 1.49806 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.35368 w_k1_ADM_s005
                                 + 1.31213 w_k1_AXP_s005
                                 + 1.11032 w_k1_DIS_s005
                                 + 1.16907 w_k1_FDX_s005 + 1.0498 w_k1_IBM_s005
                                 + 1.16499 w_k1_INTC_s005
                                 + 1.14751 w_k1_KO_s005
                                 + 0.970236 w_k1_MAT_s005
                                 + 1.15187 w_k1_MSFT_s005
                                 + 1.04522 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.902921 w_k1_ADM_s005
                                 + 0.823607 w_k1_AXP_s005
                                 + 1.03382 w_k1_DIS_s005
                                 + 0.870712 w_k1_FDX_s005
                                 + 1.26837 w_k1_IBM_s005
                                 + 0.982277 w_k1_INTC_s005
                                 + 1.36672 w_k1_KO_s005 + 1.22236 w_k1_MAT_s005
                                 + 0.959448 w_k1_MSFT_s005
                                 + 0.943396 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.02547 w_k1_ADM_s005
                                 + 1.78577 w_k1_AXP_s005
                                 + 1.81672 w_k1_DIS_s005
                                 + 1.33795 w_k1_FDX_s005
                                 + 1.70506 w_k1_IBM_s005
                                 + 1.24235 w_k1_INTC_s005
                                 + 1.57172 w_k1_KO_s005 + 2.08333 w_k1_MAT_s005
                                 + 1.20649 w_k1_MSFT_s005
                                 + 1.17867 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.0107 w_k1_ADM_s005
                                 + 0.750868 w_k1_AXP_s005
                                 + 0.790318 w_k1_DIS_s005
                                 + 0.956072 w_k1_FDX_s005
                                 + 0.555681 w_k1_IBM_s005
                                 + 0.943747 w_k1_INTC_s005
                                 + 0.765932 w_k1_KO_s005
                                 + 0.405791 w_k1_MAT_s005
                                 + 0.908024 w_k1_MSFT_s005
                                 + 0.817559 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.922775 w_k1_ADM_s005
                                 + 0.91412 w_k1_AXP_s005
                                 + 0.767936 w_k1_DIS_s005
                                 + 0.845417 w_k1_FDX_s005
                                 + 0.76216 w_k1_IBM_s005
                                 + 0.783236 w_k1_INTC_s005
                                 + 0.815774 w_k1_KO_s005
                                 + 0.876968 w_k1_MAT_s005
                                 + 0.856513 w_k1_MSFT_s005
                                 + 0.865963 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.878778 w_k1_ADM_s005
                                 + 0.769311 w_k1_AXP_s005
                                 + 0.834194 w_k1_DIS_s005
                                 + 0.835751 w_k1_FDX_s005
                                 + 0.972069 w_k1_IBM_s005
                                 + 0.752582 w_k1_INTC_s005
                                 + 0.920621 w_k1_KO_s005
                                 + 1.00318 w_k1_MAT_s005
                                 + 0.885564 w_k1_MSFT_s005
                                 + 0.833342 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.04964 w_k1_ADM_s005
                                 + 0.931157 w_k1_AXP_s005
                                 + 1.33955 w_k1_DIS_s005 + 1.2492 w_k1_FDX_s005
                                 + 0.82697 w_k1_IBM_s005
                                 + 0.898636 w_k1_INTC_s005
                                 + 1.19444 w_k1_KO_s005
                                 + 0.794502 w_k1_MAT_s005
                                 + 1.22279 w_k1_MSFT_s005
                                 + 1.04675 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.739689 w_k1_ADM_s005
                                 + 0.564066 w_k1_AXP_s005
                                 + 0.547594 w_k1_DIS_s005
                                 + 0.641223 w_k1_FDX_s005
                                 + 0.683155 w_k1_IBM_s005
                                 + 0.830599 w_k1_INTC_s005
                                 + 0.714806 w_k1_KO_s005
                                 + 0.496055 w_k1_MAT_s005
                                 + 0.76967 w_k1_MSFT_s005
                                 + 0.72051 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.698655 w_k1_ADM_s006
                                 + 0.964437 w_k1_AXP_s006
                                 + 1.05748 w_k1_DIS_s006
                                 + 0.963487 w_k1_FDX_s006
                                 + 1.12631 w_k1_IBM_s006
                                 + 1.20649 w_k1_INTC_s006
                                 + 1.06355 w_k1_KO_s006
                                 + 0.963329 w_k1_MAT_s006
                                 + 0.935275 w_k1_MSFT_s006
                                 + 0.913757 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.11705 w_k1_ADM_s006
                                 + 1.1918 w_k1_AXP_s006 + 1.43126 w_k1_DIS_s006
                                 + 1.27561 w_k1_FDX_s006
                                 + 1.35673 w_k1_IBM_s006
                                 + 1.18833 w_k1_INTC_s006
                                 + 1.41855 w_k1_KO_s006 + 1.23971 w_k1_MAT_s006
                                 + 1.25925 w_k1_MSFT_s006
                                 + 1.03623 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.870154 w_k1_ADM_s006
                                 + 1.20056 w_k1_AXP_s006
                                 + 0.935214 w_k1_DIS_s006
                                 + 0.874564 w_k1_FDX_s006
                                 + 1.68536 w_k1_IBM_s006
                                 + 0.994635 w_k1_INTC_s006
                                 + 1.17553 w_k1_KO_s006 + 1.91813 w_k1_MAT_s006
                                 + 1.01153 w_k1_MSFT_s006
                                 + 1.04035 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.917721 w_k1_ADM_s006
                                 + 0.831432 w_k1_AXP_s006
                                 + 0.884922 w_k1_DIS_s006
                                 + 0.926649 w_k1_FDX_s006
                                 + 0.680701 w_k1_IBM_s006
                                 + 0.919887 w_k1_INTC_s006
                                 + 0.673421 w_k1_KO_s006
                                 + 0.646025 w_k1_MAT_s006
                                 + 0.990853 w_k1_MSFT_s006
                                 + 0.834632 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.857846 w_k1_ADM_s006
                                 + 0.620561 w_k1_AXP_s006
                                 + 0.633203 w_k1_DIS_s006
                                 + 0.839993 w_k1_FDX_s006
                                 + 0.66598 w_k1_IBM_s006
                                 + 0.995256 w_k1_INTC_s006
                                 + 0.8116 w_k1_KO_s006 + 0.400626 w_k1_MAT_s006
                                 + 0.904655 w_k1_MSFT_s006
                                 + 0.876537 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.720491 w_k1_ADM_s006
                                 + 0.778042 w_k1_AXP_s006
                                 + 0.843792 w_k1_DIS_s006
                                 + 0.768087 w_k1_FDX_s006
                                 + 1.12978 w_k1_IBM_s006
                                 + 0.874384 w_k1_INTC_s006
                                 + 0.914217 w_k1_KO_s006 + 1.1264 w_k1_MAT_s006
                                 + 0.809335 w_k1_MSFT_s006
                                 + 0.840999 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.839244 w_k1_ADM_s006
                                 + 0.958212 w_k1_AXP_s006
                                 + 1.04823 w_k1_DIS_s006
                                 + 0.937725 w_k1_FDX_s006
                                 + 1.4903 w_k1_IBM_s006
                                 + 1.20331 w_k1_INTC_s006
                                 + 1.25712 w_k1_KO_s006 + 1.51293 w_k1_MAT_s006
                                 + 0.960597 w_k1_MSFT_s006
                                 + 0.97687 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.897478 w_k1_ADM_s006
                                 + 1.18737 w_k1_AXP_s006
                                 + 0.936187 w_k1_DIS_s006
                                 + 1.10574 w_k1_FDX_s006
                                 + 0.981452 w_k1_IBM_s006
                                 + 1.21345 w_k1_INTC_s006
                                 + 0.857424 w_k1_KO_s006
                                 + 0.879147 w_k1_MAT_s006
                                 + 0.982008 w_k1_MSFT_s006
                                 + 1.02936 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.12511 w_k1_ADM_s007
                                 + 1.31392 w_k1_AXP_s007
                                 + 1.18625 w_k1_DIS_s007
                                 + 0.97985 w_k1_FDX_s007 + 1.5117 w_k1_IBM_s007
                                 + 1.22101 w_k1_INTC_s007
                                 + 1.28099 w_k1_KO_s007 + 1.65374 w_k1_MAT_s007
                                 + 1.1118 w_k1_MSFT_s007 + 1.08227 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.15299 w_k1_ADM_s007
                                 + 1.15319 w_k1_AXP_s007
                                 + 1.10233 w_k1_DIS_s007
                                 + 1.10319 w_k1_FDX_s007
                                 + 1.05721 w_k1_IBM_s007
                                 + 1.16968 w_k1_INTC_s007
                                 + 1.04984 w_k1_KO_s007 + 1.40274 w_k1_MAT_s007
                                 + 1.14707 w_k1_MSFT_s007
                                 + 1.04406 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.10499 w_k1_ADM_s007
                                 + 0.80023 w_k1_AXP_s007
                                 + 0.90233 w_k1_DIS_s007
                                 + 0.926704 w_k1_FDX_s007
                                 + 0.751375 w_k1_IBM_s007
                                 + 1.07172 w_k1_INTC_s007
                                 + 0.961209 w_k1_KO_s007
                                 + 0.812731 w_k1_MAT_s007
                                 + 1.09807 w_k1_MSFT_s007
                                 + 1.05167 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.993172 w_k1_ADM_s007
                                 + 1.14219 w_k1_AXP_s007
                                 + 1.06634 w_k1_DIS_s007
                                 + 1.11269 w_k1_FDX_s007
                                 + 0.791182 w_k1_IBM_s007
                                 + 1.23099 w_k1_INTC_s007
                                 + 0.888084 w_k1_KO_s007
                                 + 0.970238 w_k1_MAT_s007
                                 + 1.05034 w_k1_MSFT_s007
                                 + 0.945468 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.961904 w_k1_ADM_s007
                                 + 0.851168 w_k1_AXP_s007
                                 + 0.983099 w_k1_DIS_s007
                                 + 0.893093 w_k1_FDX_s007
                                 + 1.15621 w_k1_IBM_s007
                                 + 1.04421 w_k1_INTC_s007
                                 + 1.01287 w_k1_KO_s007 + 1.27117 w_k1_MAT_s007
                                 + 0.943979 w_k1_MSFT_s007
                                 + 1.00843 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04079 w_k1_ADM_s007
                                 + 1.38671 w_k1_AXP_s007
                                 + 1.48621 w_k1_DIS_s007
                                 + 1.25102 w_k1_FDX_s007
                                 + 0.724947 w_k1_IBM_s007
                                 + 1.21405 w_k1_INTC_s007
                                 + 1.08554 w_k1_KO_s007 + 1.14933 w_k1_MAT_s007
                                 + 1.22761 w_k1_MSFT_s007
                                 + 1.16324 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.857227 w_k1_ADM_s007
                                 + 0.604273 w_k1_AXP_s007
                                 + 0.689175 w_k1_DIS_s007
                                 + 0.746471 w_k1_FDX_s007
                                 + 0.805366 w_k1_IBM_s007
                                 + 0.889686 w_k1_INTC_s007
                                 + 0.75864 w_k1_KO_s007
                                 + 0.803932 w_k1_MAT_s007
                                 + 0.804829 w_k1_MSFT_s007
                                 + 0.859504 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.13633 w_k1_ADM_s007
                                 + 1.39725 w_k1_AXP_s007
                                 + 1.59727 w_k1_DIS_s007
                                 + 1.35683 w_k1_FDX_s007 + 1.1447 w_k1_IBM_s007
                                 + 1.28242 w_k1_INTC_s007
                                 + 1.22213 w_k1_KO_s007 + 1.55553 w_k1_MAT_s007
                                 + 1.39206 w_k1_MSFT_s007 + 1.3151 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.14085 w_k1_ADM_s010
                                 + 1.22607 w_k1_AXP_s010
                                 + 1.56792 w_k1_DIS_s010
                                 + 1.40696 w_k1_FDX_s010
                                 + 0.94752 w_k1_IBM_s010
                                 + 1.18682 w_k1_INTC_s010 + 1.2442 w_k1_KO_s010
                                 + 0.995638 w_k1_MAT_s010
                                 + 1.10411 w_k1_MSFT_s010
                                 + 1.18686 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.900809 w_k1_ADM_s010
                                 + 0.755083 w_k1_AXP_s010
                                 + 1.04673 w_k1_DIS_s010
                                 + 1.15124 w_k1_FDX_s010
                                 + 0.612749 w_k1_IBM_s010
                                 + 1.04116 w_k1_INTC_s010
                                 + 1.01544 w_k1_KO_s010
                                 + 0.748565 w_k1_MAT_s010
                                 + 1.05067 w_k1_MSFT_s010
                                 + 0.899612 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.893134 w_k1_ADM_s010
                                 + 0.661622 w_k1_AXP_s010
                                 + 0.638883 w_k1_DIS_s010
                                 + 0.799836 w_k1_FDX_s010
                                 + 1.02168 w_k1_IBM_s010
                                 + 0.913096 w_k1_INTC_s010
                                 + 0.889211 w_k1_KO_s010
                                 + 0.898228 w_k1_MAT_s010
                                 + 0.720612 w_k1_MSFT_s010
                                 + 0.906779 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.792983 w_k1_ADM_s010
                                 + 0.628049 w_k1_AXP_s010
                                 + 0.71279 w_k1_DIS_s010
                                 + 0.849327 w_k1_FDX_s010
                                 + 0.653028 w_k1_IBM_s010
                                 + 0.718811 w_k1_INTC_s010
                                 + 0.681372 w_k1_KO_s010
                                 + 0.56162 w_k1_MAT_s010
                                 + 0.736895 w_k1_MSFT_s010
                                 + 0.852492 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.08598 w_k1_ADM_s010
                                 + 1.24474 w_k1_AXP_s010
                                 + 1.02066 w_k1_DIS_s010 + 1.0707 w_k1_FDX_s010
                                 + 1.39053 w_k1_IBM_s010
                                 + 1.15307 w_k1_INTC_s010
                                 + 1.16968 w_k1_KO_s010 + 1.07724 w_k1_MAT_s010
                                 + 0.841898 w_k1_MSFT_s010
                                 + 1.18118 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.18021 w_k1_ADM_s010
                                 + 1.10319 w_k1_AXP_s010
                                 + 1.08808 w_k1_DIS_s010
                                 + 1.25334 w_k1_FDX_s010
                                 + 0.587562 w_k1_IBM_s010
                                 + 1.04312 w_k1_INTC_s010
                                 + 0.806568 w_k1_KO_s010
                                 + 0.508944 w_k1_MAT_s010
                                 + 0.923681 w_k1_MSFT_s010
                                 + 1.00971 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.852026 w_k1_ADM_s010
                                 + 0.755853 w_k1_AXP_s010
                                 + 0.913411 w_k1_DIS_s010
                                 + 0.931877 w_k1_FDX_s010
                                 + 0.928363 w_k1_IBM_s010
                                 + 1.03516 w_k1_INTC_s010
                                 + 1.01711 w_k1_KO_s010
                                 + 0.670635 w_k1_MAT_s010
                                 + 0.982006 w_k1_MSFT_s010 + 1.072 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.849542 w_k1_ADM_s010
                                 + 0.8628 w_k1_AXP_s010 + 0.61924 w_k1_DIS_s010
                                 + 0.800944 w_k1_FDX_s010
                                 + 1.31203 w_k1_IBM_s010
                                 + 0.905683 w_k1_INTC_s010
                                 + 0.955629 w_k1_KO_s010
                                 + 1.47948 w_k1_MAT_s010
                                 + 0.737604 w_k1_MSFT_s010
                                 + 0.937243 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.835792 w_k1_ADM_s011
                                 + 0.731078 w_k1_AXP_s011
                                 + 0.770479 w_k1_DIS_s011
                                 + 0.859359 w_k1_FDX_s011
                                 + 0.659512 w_k1_IBM_s011
                                 + 0.889037 w_k1_INTC_s011
                                 + 0.742355 w_k1_KO_s011
                                 + 0.509038 w_k1_MAT_s011
                                 + 1.00195 w_k1_MSFT_s011
                                 + 0.796199 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.884524 w_k1_ADM_s011
                                 + 1.35835 w_k1_AXP_s011 + 1.0082 w_k1_DIS_s011
                                 + 0.998098 w_k1_FDX_s011
                                 + 1.49271 w_k1_IBM_s011
                                 + 0.896547 w_k1_INTC_s011
                                 + 1.01873 w_k1_KO_s011 + 1.80948 w_k1_MAT_s011
                                 + 0.909714 w_k1_MSFT_s011
                                 + 0.986508 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.812386 w_k1_ADM_s011
                                 + 0.806752 w_k1_AXP_s011
                                 + 0.886069 w_k1_DIS_s011
                                 + 0.80206 w_k1_FDX_s011
                                 + 1.33545 w_k1_IBM_s011
                                 + 0.860624 w_k1_INTC_s011
                                 + 1.00873 w_k1_KO_s011 + 1.13601 w_k1_MAT_s011
                                 + 0.733693 w_k1_MSFT_s011
                                 + 0.834145 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.739181 w_k1_ADM_s011
                                 + 0.936772 w_k1_AXP_s011
                                 + 0.625262 w_k1_DIS_s011
                                 + 0.841485 w_k1_FDX_s011
                                 + 0.937225 w_k1_IBM_s011
                                 + 0.850334 w_k1_INTC_s011
                                 + 0.720151 w_k1_KO_s011
                                 + 1.17104 w_k1_MAT_s011
                                 + 0.738382 w_k1_MSFT_s011
                                 + 0.824452 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.892052 w_k1_ADM_s011
                                 + 0.924108 w_k1_AXP_s011
                                 + 0.917268 w_k1_DIS_s011
                                 + 0.967455 w_k1_FDX_s011
                                 + 0.964045 w_k1_IBM_s011
                                 + 1.05115 w_k1_INTC_s011
                                 + 0.836616 w_k1_KO_s011
                                 + 0.875135 w_k1_MAT_s011
                                 + 0.926677 w_k1_MSFT_s011
                                 + 0.814605 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.938451 w_k1_ADM_s011
                                 + 1.33298 w_k1_AXP_s011
                                 + 1.77559 w_k1_DIS_s011
                                 + 1.33765 w_k1_FDX_s011
                                 + 1.42389 w_k1_IBM_s011
                                 + 1.12329 w_k1_INTC_s011
                                 + 1.25733 w_k1_KO_s011 + 1.68252 w_k1_MAT_s011
                                 + 1.21485 w_k1_MSFT_s011
                                 + 1.19573 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.09049 w_k1_ADM_s011
                                 + 0.885597 w_k1_AXP_s011
                                 + 1.30876 w_k1_DIS_s011
                                 + 1.17621 w_k1_FDX_s011
                                 + 0.789899 w_k1_IBM_s011
                                 + 1.11503 w_k1_INTC_s011
                                 + 0.968695 w_k1_KO_s011
                                 + 0.615899 w_k1_MAT_s011
                                 + 1.24275 w_k1_MSFT_s011
                                 + 0.981713 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.06797 w_k1_ADM_s011
                                 + 1.33453 w_k1_AXP_s011
                                 + 1.13429 w_k1_DIS_s011
                                 + 1.28737 w_k1_FDX_s011
                                 + 0.956859 w_k1_IBM_s011
                                 + 1.17581 w_k1_INTC_s011
                                 + 0.873049 w_k1_KO_s011
                                 + 1.26046 w_k1_MAT_s011
                                 + 1.0272 w_k1_MSFT_s011 + 1.01912 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.926536 w_k1_ADM_s012
                                 + 0.92066 w_k1_AXP_s012
                                 + 0.643681 w_k1_DIS_s012
                                 + 0.796862 w_k1_FDX_s012
                                 + 0.764242 w_k1_IBM_s012
                                 + 0.8036 w_k1_INTC_s012
                                 + 0.779859 w_k1_KO_s012
                                 + 0.768995 w_k1_MAT_s012
                                 + 0.771438 w_k1_MSFT_s012
                                 + 0.801674 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.908751 w_k1_ADM_s012
                                 + 1.31689 w_k1_AXP_s012
                                 + 0.943215 w_k1_DIS_s012
                                 + 0.954315 w_k1_FDX_s012
                                 + 0.972778 w_k1_IBM_s012
                                 + 0.910845 w_k1_INTC_s012
                                 + 0.919046 w_k1_KO_s012
                                 + 1.33145 w_k1_MAT_s012
                                 + 0.859922 w_k1_MSFT_s012
                                 + 0.951241 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.957758 w_k1_ADM_s012
                                 + 1.34177 w_k1_AXP_s012
                                 + 0.951841 w_k1_DIS_s012
                                 + 1.01218 w_k1_FDX_s012 + 1.2418 w_k1_IBM_s012
                                 + 1.0332 w_k1_INTC_s012 + 1.09142 w_k1_KO_s012
                                 + 1.24027 w_k1_MAT_s012
                                 + 0.869375 w_k1_MSFT_s012
                                 + 1.00112 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.925658 w_k1_ADM_s012
                                 + 0.892578 w_k1_AXP_s012
                                 + 0.930083 w_k1_DIS_s012
                                 + 0.864998 w_k1_FDX_s012
                                 + 1.12541 w_k1_IBM_s012
                                 + 0.962494 w_k1_INTC_s012
                                 + 1.16818 w_k1_KO_s012
                                 + 0.874534 w_k1_MAT_s012
                                 + 1.01089 w_k1_MSFT_s012
                                 + 0.984798 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.851379 w_k1_ADM_s012
                                 + 1.31355 w_k1_AXP_s012
                                 + 1.15903 w_k1_DIS_s012
                                 + 0.966233 w_k1_FDX_s012
                                 + 1.67046 w_k1_IBM_s012
                                 + 0.953909 w_k1_INTC_s012
                                 + 1.14223 w_k1_KO_s012 + 1.60676 w_k1_MAT_s012
                                 + 0.928691 w_k1_MSFT_s012
                                 + 1.02789 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.00766 w_k1_ADM_s012
                                 + 1.03967 w_k1_AXP_s012
                                 + 0.994011 w_k1_DIS_s012
                                 + 1.08721 w_k1_FDX_s012
                                 + 0.658046 w_k1_IBM_s012
                                 + 0.839447 w_k1_INTC_s012
                                 + 0.977778 w_k1_KO_s012
                                 + 0.763097 w_k1_MAT_s012
                                 + 0.982146 w_k1_MSFT_s012
                                 + 1.02731 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.39981 w_k1_ADM_s012
                                 + 1.70366 w_k1_AXP_s012
                                 + 1.66357 w_k1_DIS_s012 + 1.3505 w_k1_FDX_s012
                                 + 1.01593 w_k1_IBM_s012
                                 + 1.33571 w_k1_INTC_s012
                                 + 1.38988 w_k1_KO_s012 + 0.98862 w_k1_MAT_s012
                                 + 1.15255 w_k1_MSFT_s012
                                 + 1.13317 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.927553 w_k1_ADM_s012
                                 + 1.25343 w_k1_AXP_s012
                                 + 0.998762 w_k1_DIS_s012
                                 + 1.11383 w_k1_FDX_s012
                                 + 0.925306 w_k1_IBM_s012
                                 + 1.02485 w_k1_INTC_s012
                                 + 1.02708 w_k1_KO_s012
                                 + 0.879141 w_k1_MAT_s012
                                 + 0.951371 w_k1_MSFT_s012
                                 + 0.967404 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.800766 w_k1_ADM_s013
                                 + 1.13998 w_k1_AXP_s013
                                 + 1.16922 w_k1_DIS_s013
                                 + 0.934349 w_k1_FDX_s013
                                 + 1.74621 w_k1_IBM_s013
                                 + 1.04745 w_k1_INTC_s013
                                 + 1.36982 w_k1_KO_s013 + 1.38295 w_k1_MAT_s013
                                 + 1.03235 w_k1_MSFT_s013
                                 + 1.13652 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.95302 w_k1_ADM_s013
                                 + 1.0248 w_k1_AXP_s013 + 1.40438 w_k1_DIS_s013
                                 + 1.16409 w_k1_FDX_s013
                                 + 1.02007 w_k1_IBM_s013
                                 + 1.27221 w_k1_INTC_s013
                                 + 1.25972 w_k1_KO_s013
                                 + 0.806067 w_k1_MAT_s013
                                 + 1.38317 w_k1_MSFT_s013
                                 + 1.19455 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.71455 w_k1_ADM_s013
                                 + 0.958286 w_k1_AXP_s013
                                 + 0.802385 w_k1_DIS_s013
                                 + 0.866637 w_k1_FDX_s013
                                 + 1.14528 w_k1_IBM_s013
                                 + 0.899383 w_k1_INTC_s013
                                 + 0.961745 w_k1_KO_s013
                                 + 1.23717 w_k1_MAT_s013
                                 + 0.986483 w_k1_MSFT_s013
                                 + 1.01331 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.28333 w_k1_ADM_s013
                                 + 1.62318 w_k1_AXP_s013
                                 + 1.47296 w_k1_DIS_s013
                                 + 1.38032 w_k1_FDX_s013
                                 + 1.22562 w_k1_IBM_s013
                                 + 1.30693 w_k1_INTC_s013
                                 + 1.23103 w_k1_KO_s013 + 1.17375 w_k1_MAT_s013
                                 + 1.47265 w_k1_MSFT_s013
                                 + 1.27377 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.06117 w_k1_ADM_s013
                                 + 1.29977 w_k1_AXP_s013
                                 + 1.15974 w_k1_DIS_s013 + 1.2368 w_k1_FDX_s013
                                 + 1.25651 w_k1_IBM_s013
                                 + 1.0522 w_k1_INTC_s013 + 1.17988 w_k1_KO_s013
                                 + 1.06069 w_k1_MAT_s013
                                 + 1.05239 w_k1_MSFT_s013
                                 + 1.17093 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.76294 w_k1_ADM_s013
                                 + 0.733699 w_k1_AXP_s013
                                 + 0.641708 w_k1_DIS_s013
                                 + 0.794081 w_k1_FDX_s013
                                 + 0.817183 w_k1_IBM_s013
                                 + 1.08407 w_k1_INTC_s013
                                 + 0.873257 w_k1_KO_s013
                                 + 0.565425 w_k1_MAT_s013
                                 + 0.866652 w_k1_MSFT_s013
                                 + 0.853465 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.0336 w_k1_ADM_s013
                                 + 1.12384 w_k1_AXP_s013
                                 + 0.995511 w_k1_DIS_s013
                                 + 1.0529 w_k1_FDX_s013 + 1.28227 w_k1_IBM_s013
                                 + 1.03487 w_k1_INTC_s013
                                 + 1.17909 w_k1_KO_s013
                                 + 0.876275 w_k1_MAT_s013
                                 + 1.08765 w_k1_MSFT_s013
                                 + 0.969639 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.935491 w_k1_ADM_s013
                                 + 1.01404 w_k1_AXP_s013
                                 + 0.96355 w_k1_DIS_s013 + 1.0245 w_k1_FDX_s013
                                 + 0.810238 w_k1_IBM_s013
                                 + 1.06263 w_k1_INTC_s013
                                 + 0.869635 w_k1_KO_s013
                                 + 0.623784 w_k1_MAT_s013
                                 + 0.930603 w_k1_MSFT_s013
                                 + 0.954853 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.984794 w_k1_ADM_s014
                                 + 0.972222 w_k1_AXP_s014
                                 + 0.769275 w_k1_DIS_s014
                                 + 0.975165 w_k1_FDX_s014
                                 + 0.71278 w_k1_IBM_s014
                                 + 0.772418 w_k1_INTC_s014
                                 + 0.642195 w_k1_KO_s014
                                 + 0.682733 w_k1_MAT_s014
                                 + 0.87071 w_k1_MSFT_s014
                                 + 0.904797 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.53552 w_k1_ADM_s014
                                 + 1.45349 w_k1_AXP_s014
                                 + 1.80164 w_k1_DIS_s014
                                 + 1.40124 w_k1_FDX_s014
                                 + 1.60506 w_k1_IBM_s014
                                 + 1.37254 w_k1_INTC_s014
                                 + 1.26659 w_k1_KO_s014 + 1.01345 w_k1_MAT_s014
                                 + 1.35771 w_k1_MSFT_s014
                                 + 1.42015 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.710407 w_k1_ADM_s014
                                 + 0.647317 w_k1_AXP_s014
                                 + 0.424123 w_k1_DIS_s014
                                 + 0.679876 w_k1_FDX_s014
                                 + 0.689657 w_k1_IBM_s014
                                 + 0.735296 w_k1_INTC_s014
                                 + 0.551869 w_k1_KO_s014
                                 + 0.532551 w_k1_MAT_s014
                                 + 0.79361 w_k1_MSFT_s014
                                 + 0.790866 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.849913 w_k1_ADM_s014
                                 + 0.959242 w_k1_AXP_s014
                                 + 0.877282 w_k1_DIS_s014
                                 + 0.867621 w_k1_FDX_s014
                                 + 1.43591 w_k1_IBM_s014
                                 + 0.821114 w_k1_INTC_s014
                                 + 1.00514 w_k1_KO_s014 + 1.26603 w_k1_MAT_s014
                                 + 0.824593 w_k1_MSFT_s014
                                 + 1.01091 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.10945 w_k1_ADM_s014
                                 + 1.29591 w_k1_AXP_s014
                                 + 1.11517 w_k1_DIS_s014
                                 + 1.13546 w_k1_FDX_s014
                                 + 1.06463 w_k1_IBM_s014
                                 + 0.882487 w_k1_INTC_s014
                                 + 0.931654 w_k1_KO_s014
                                 + 1.19964 w_k1_MAT_s014
                                 + 0.95143 w_k1_MSFT_s014 + 1.0358 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.30357 w_k1_ADM_s014
                                 + 1.3539 w_k1_AXP_s014 + 1.37613 w_k1_DIS_s014
                                 + 1.39349 w_k1_FDX_s014
                                 + 0.97694 w_k1_IBM_s014
                                 + 1.03356 w_k1_INTC_s014 + 1.0469 w_k1_KO_s014
                                 + 1.08581 w_k1_MAT_s014
                                 + 1.11452 w_k1_MSFT_s014
                                 + 1.26895 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.885409 w_k1_ADM_s014
                                 + 1.01037 w_k1_AXP_s014
                                 + 1.06568 w_k1_DIS_s014
                                 + 0.967256 w_k1_FDX_s014
                                 + 1.25893 w_k1_IBM_s014
                                 + 1.10764 w_k1_INTC_s014
                                 + 0.97049 w_k1_KO_s014 + 1.39486 w_k1_MAT_s014
                                 + 1.06747 w_k1_MSFT_s014
                                 + 1.05171 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.850846 w_k1_ADM_s014
                                 + 0.864866 w_k1_AXP_s014
                                 + 0.830051 w_k1_DIS_s014
                                 + 0.888419 w_k1_FDX_s014
                                 + 0.922497 w_k1_IBM_s014
                                 + 0.962859 w_k1_INTC_s014
                                 + 0.78115 w_k1_KO_s014
                                 + 0.748512 w_k1_MAT_s014
                                 + 0.991556 w_k1_MSFT_s014
                                 + 0.943777 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.905335 w_k1_ADM_s015
                                 + 0.83432 w_k1_AXP_s015
                                 + 0.984915 w_k1_DIS_s015
                                 + 1.04808 w_k1_FDX_s015
                                 + 0.758927 w_k1_IBM_s015
                                 + 0.923581 w_k1_INTC_s015
                                 + 0.893192 w_k1_KO_s015 + 0.8377 w_k1_MAT_s015
                                 + 1.0334 w_k1_MSFT_s015 + 1.01366 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.927067 w_k1_ADM_s015
                                 + 1.01384 w_k1_AXP_s015
                                 + 1.05033 w_k1_DIS_s015
                                 + 0.968192 w_k1_FDX_s015
                                 + 1.33163 w_k1_IBM_s015
                                 + 0.917266 w_k1_INTC_s015
                                 + 1.21217 w_k1_KO_s015 + 1.6126 w_k1_MAT_s015
                                 + 1.03135 w_k1_MSFT_s015
                                 + 0.956156 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.40994 w_k1_ADM_s015
                                 + 1.58279 w_k1_AXP_s015
                                 + 1.53962 w_k1_DIS_s015
                                 + 1.30133 w_k1_FDX_s015
                                 + 1.54233 w_k1_IBM_s015
                                 + 1.4329 w_k1_INTC_s015 + 1.48627 w_k1_KO_s015
                                 + 1.7301 w_k1_MAT_s015
                                 + 1.31405 w_k1_MSFT_s015
                                 + 1.34264 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.928691 w_k1_ADM_s015
                                 + 0.789525 w_k1_AXP_s015
                                 + 0.768751 w_k1_DIS_s015
                                 + 0.885416 w_k1_FDX_s015
                                 + 0.723343 w_k1_IBM_s015
                                 + 0.856963 w_k1_INTC_s015
                                 + 0.784736 w_k1_KO_s015
                                 + 0.810701 w_k1_MAT_s015
                                 + 0.903189 w_k1_MSFT_s015
                                 + 0.843671 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.10734 w_k1_ADM_s015
                                 + 1.43781 w_k1_AXP_s015
                                 + 1.31464 w_k1_DIS_s015
                                 + 1.29258 w_k1_FDX_s015
                                 + 0.979987 w_k1_IBM_s015
                                 + 1.11585 w_k1_INTC_s015
                                 + 1.14894 w_k1_KO_s015 + 1.15183 w_k1_MAT_s015
                                 + 1.41443 w_k1_MSFT_s015
                                 + 1.13494 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.855968 w_k1_ADM_s015
                                 + 1.18866 w_k1_AXP_s015
                                 + 1.04249 w_k1_DIS_s015
                                 + 1.02072 w_k1_FDX_s015
                                 + 0.974812 w_k1_IBM_s015
                                 + 1.14746 w_k1_INTC_s015
                                 + 1.15764 w_k1_KO_s015 + 1.17838 w_k1_MAT_s015
                                 + 1.04839 w_k1_MSFT_s015
                                 + 1.03349 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.991737 w_k1_ADM_s015
                                 + 1.03938 w_k1_AXP_s015
                                 + 0.678179 w_k1_DIS_s015
                                 + 0.868575 w_k1_FDX_s015
                                 + 1.05279 w_k1_IBM_s015
                                 + 1.19948 w_k1_INTC_s015
                                 + 0.908096 w_k1_KO_s015
                                 + 1.20667 w_k1_MAT_s015
                                 + 0.934872 w_k1_MSFT_s015
                                 + 1.06654 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.710522 w_k1_ADM_s015
                                 + 0.492236 w_k1_AXP_s015
                                 + 0.799462 w_k1_DIS_s015
                                 + 0.781349 w_k1_FDX_s015
                                 + 0.821463 w_k1_IBM_s015
                                 + 0.767982 w_k1_INTC_s015
                                 + 0.933769 w_k1_KO_s015
                                 + 0.834577 w_k1_MAT_s015
                                 + 0.95816 w_k1_MSFT_s015
                                 + 0.936554 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.959192 w_k1_ADM_s016
                                 + 1.27922 w_k1_AXP_s016 + 1.2128 w_k1_DIS_s016
                                 + 0.999261 w_k1_FDX_s016
                                 + 1.62967 w_k1_IBM_s016
                                 + 0.994424 w_k1_INTC_s016
                                 + 1.33494 w_k1_KO_s016 + 2.20369 w_k1_MAT_s016
                                 + 0.947678 w_k1_MSFT_s016
                                 + 1.19328 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.923688 w_k1_ADM_s016
                                 + 0.847383 w_k1_AXP_s016
                                 + 0.894851 w_k1_DIS_s016
                                 + 0.943187 w_k1_FDX_s016
                                 + 1.06835 w_k1_IBM_s016
                                 + 0.851925 w_k1_INTC_s016
                                 + 1.00026 w_k1_KO_s016 + 1.61189 w_k1_MAT_s016
                                 + 0.864304 w_k1_MSFT_s016
                                 + 1.00419 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.14657 w_k1_ADM_s016
                                 + 1.16416 w_k1_AXP_s016
                                 + 1.06377 w_k1_DIS_s016
                                 + 1.05263 w_k1_FDX_s016
                                 + 1.08876 w_k1_IBM_s016
                                 + 0.981002 w_k1_INTC_s016
                                 + 1.10024 w_k1_KO_s016 + 1.77938 w_k1_MAT_s016
                                 + 0.974033 w_k1_MSFT_s016
                                 + 1.01406 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20662 w_k1_ADM_s016
                                 + 1.05535 w_k1_AXP_s016
                                 + 1.43843 w_k1_DIS_s016
                                 + 1.18356 w_k1_FDX_s016
                                 + 1.40032 w_k1_IBM_s016
                                 + 1.20462 w_k1_INTC_s016
                                 + 1.30339 w_k1_KO_s016 + 1.23001 w_k1_MAT_s016
                                 + 1.11553 w_k1_MSFT_s016
                                 + 1.12062 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.02052 w_k1_ADM_s016
                                 + 1.0395 w_k1_AXP_s016 + 1.32759 w_k1_DIS_s016
                                 + 1.21685 w_k1_FDX_s016
                                 + 0.84548 w_k1_IBM_s016
                                 + 1.1471 w_k1_INTC_s016
                                 + 0.987589 w_k1_KO_s016
                                 + 0.841048 w_k1_MAT_s016
                                 + 1.13806 w_k1_MSFT_s016
                                 + 1.01401 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.851996 w_k1_ADM_s016
                                 + 1.13048 w_k1_AXP_s016
                                 + 0.733037 w_k1_DIS_s016
                                 + 0.888579 w_k1_FDX_s016
                                 + 0.926121 w_k1_IBM_s016
                                 + 0.778594 w_k1_INTC_s016
                                 + 0.848279 w_k1_KO_s016
                                 + 1.38982 w_k1_MAT_s016
                                 + 0.778219 w_k1_MSFT_s016
                                 + 0.800466 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.838972 w_k1_ADM_s016
                                 + 0.738451 w_k1_AXP_s016
                                 + 0.699308 w_k1_DIS_s016
                                 + 0.779118 w_k1_FDX_s016
                                 + 0.793731 w_k1_IBM_s016
                                 + 0.757463 w_k1_INTC_s016
                                 + 0.760624 w_k1_KO_s016
                                 + 0.727209 w_k1_MAT_s016
                                 + 0.716123 w_k1_MSFT_s016
                                 + 0.872478 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.53559 w_k1_ADM_s016
                                 + 1.60048 w_k1_AXP_s016
                                 + 1.83898 w_k1_DIS_s016
                                 + 1.50998 w_k1_FDX_s016
                                 + 1.32108 w_k1_IBM_s016
                                 + 1.36533 w_k1_INTC_s016
                                 + 1.46254 w_k1_KO_s016 + 1.73632 w_k1_MAT_s016
                                 + 1.33404 w_k1_MSFT_s016
                                 + 1.33486 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.16415 w_k1_ADM_s017
                                 + 1.12735 w_k1_AXP_s017
                                 + 1.19208 w_k1_DIS_s017
                                 + 1.17693 w_k1_FDX_s017
                                 + 1.06655 w_k1_IBM_s017
                                 + 0.878878 w_k1_INTC_s017
                                 + 0.873004 w_k1_KO_s017
                                 + 0.999269 w_k1_MAT_s017
                                 + 0.96494 w_k1_MSFT_s017
                                 + 1.31139 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.48179 w_k1_ADM_s017
                                 + 1.58545 w_k1_AXP_s017
                                 + 1.77714 w_k1_DIS_s017
                                 + 1.60753 w_k1_FDX_s017 + 1.3213 w_k1_IBM_s017
                                 + 1.22339 w_k1_INTC_s017
                                 + 1.29577 w_k1_KO_s017 + 2.00287 w_k1_MAT_s017
                                 + 1.3649 w_k1_MSFT_s017 + 1.73521 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.67487 w_k1_ADM_s017
                                 + 0.657897 w_k1_AXP_s017
                                 + 0.609246 w_k1_DIS_s017
                                 + 0.757504 w_k1_FDX_s017
                                 + 0.81881 w_k1_IBM_s017
                                 + 0.801603 w_k1_INTC_s017
                                 + 0.657887 w_k1_KO_s017
                                 + 1.03033 w_k1_MAT_s017
                                 + 0.71724 w_k1_MSFT_s017
                                 + 0.882499 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.871273 w_k1_ADM_s017
                                 + 1.06856 w_k1_AXP_s017
                                 + 1.37177 w_k1_DIS_s017 + 1.0912 w_k1_FDX_s017
                                 + 1.30553 w_k1_IBM_s017
                                 + 0.930574 w_k1_INTC_s017
                                 + 1.15942 w_k1_KO_s017 + 1.69916 w_k1_MAT_s017
                                 + 0.96879 w_k1_MSFT_s017
                                 + 1.23284 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.834097 w_k1_ADM_s017
                                 + 0.746503 w_k1_AXP_s017
                                 + 0.864661 w_k1_DIS_s017
                                 + 0.865341 w_k1_FDX_s017
                                 + 1.21494 w_k1_IBM_s017
                                 + 0.917884 w_k1_INTC_s017
                                 + 0.993179 w_k1_KO_s017
                                 + 0.972432 w_k1_MAT_s017
                                 + 0.851542 w_k1_MSFT_s017
                                 + 0.976035 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09963 w_k1_ADM_s017
                                 + 1.24947 w_k1_AXP_s017
                                 + 0.918093 w_k1_DIS_s017
                                 + 1.02548 w_k1_FDX_s017
                                 + 0.92863 w_k1_IBM_s017
                                 + 0.974885 w_k1_INTC_s017
                                 + 0.856337 w_k1_KO_s017
                                 + 1.42355 w_k1_MAT_s017
                                 + 0.862286 w_k1_MSFT_s017
                                 + 1.0883 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.566131 w_k1_ADM_s017
                                 + 0.473286 w_k1_AXP_s017
                                 + 0.507821 w_k1_DIS_s017
                                 + 0.726951 w_k1_FDX_s017
                                 + 0.507756 w_k1_IBM_s017
                                 + 0.786723 w_k1_INTC_s017
                                 + 0.579625 w_k1_KO_s017
                                 + 0.452811 w_k1_MAT_s017
                                 + 0.665306 w_k1_MSFT_s017
                                 + 0.807385 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.859959 w_k1_ADM_s017
                                 + 0.794172 w_k1_AXP_s017
                                 + 0.890851 w_k1_DIS_s017
                                 + 0.932068 w_k1_FDX_s017
                                 + 0.849472 w_k1_IBM_s017
                                 + 0.84216 w_k1_INTC_s017
                                 + 0.762242 w_k1_KO_s017
                                 + 0.749127 w_k1_MAT_s017
                                 + 0.852663 w_k1_MSFT_s017
                                 + 0.926465 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.05089 w_k1_ADM_s100
                                 + 1.25357 w_k1_AXP_s100
                                 + 1.01434 w_k1_DIS_s100
                                 + 1.08317 w_k1_FDX_s100
                                 + 1.05539 w_k1_IBM_s100
                                 + 1.0146 w_k1_INTC_s100 + 1.0466 w_k1_KO_s100
                                 + 1.35169 w_k1_MAT_s100
                                 + 1.05851 w_k1_MSFT_s100
                                 + 0.976223 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14906 w_k1_ADM_s100
                                 + 1.27037 w_k1_AXP_s100
                                 + 1.27879 w_k1_DIS_s100
                                 + 1.28589 w_k1_FDX_s100
                                 + 0.968374 w_k1_IBM_s100
                                 + 1.30717 w_k1_INTC_s100
                                 + 1.06483 w_k1_KO_s100
                                 + 0.743254 w_k1_MAT_s100
                                 + 1.17561 w_k1_MSFT_s100
                                 + 1.05602 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.981407 w_k1_ADM_s100
                                 + 0.795918 w_k1_AXP_s100
                                 + 0.731735 w_k1_DIS_s100
                                 + 0.859122 w_k1_FDX_s100
                                 + 0.894909 w_k1_IBM_s100
                                 + 0.885017 w_k1_INTC_s100
                                 + 0.829716 w_k1_KO_s100
                                 + 0.739979 w_k1_MAT_s100
                                 + 0.968821 w_k1_MSFT_s100
                                 + 0.929579 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.993688 w_k1_ADM_s100
                                 + 0.844777 w_k1_AXP_s100
                                 + 1.12204 w_k1_DIS_s100
                                 + 0.974121 w_k1_FDX_s100
                                 + 1.02877 w_k1_IBM_s100
                                 + 1.16013 w_k1_INTC_s100
                                 + 1.07175 w_k1_KO_s100
                                 + 0.869901 w_k1_MAT_s100
                                 + 1.12759 w_k1_MSFT_s100
                                 + 1.07041 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.08419 w_k1_ADM_s100
                                 + 0.990747 w_k1_AXP_s100
                                 + 0.943568 w_k1_DIS_s100
                                 + 1.0221 w_k1_FDX_s100
                                 + 0.621068 w_k1_IBM_s100
                                 + 1.01567 w_k1_INTC_s100
                                 + 0.812352 w_k1_KO_s100
                                 + 0.76425 w_k1_MAT_s100
                                 + 1.08407 w_k1_MSFT_s100
                                 + 0.855496 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.24613 w_k1_ADM_s100
                                 + 1.41122 w_k1_AXP_s100
                                 + 1.71898 w_k1_DIS_s100
                                 + 1.32416 w_k1_FDX_s100
                                 + 1.51032 w_k1_IBM_s100
                                 + 1.40261 w_k1_INTC_s100
                                 + 1.42851 w_k1_KO_s100 + 1.49614 w_k1_MAT_s100
                                 + 1.32941 w_k1_MSFT_s100
                                 + 1.20872 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.816342 w_k1_ADM_s100
                                 + 0.889915 w_k1_AXP_s100
                                 + 0.679286 w_k1_DIS_s100
                                 + 0.777724 w_k1_FDX_s100
                                 + 1.12495 w_k1_IBM_s100
                                 + 0.796261 w_k1_INTC_s100
                                 + 0.873441 w_k1_KO_s100
                                 + 1.32683 w_k1_MAT_s100
                                 + 0.753188 w_k1_MSFT_s100
                                 + 0.87014 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.1629 w_k1_ADM_s100
                                 + 1.19822 w_k1_AXP_s100
                                 + 1.28482 w_k1_DIS_s100
                                 + 1.20368 w_k1_FDX_s100
                                 + 1.26532 w_k1_IBM_s100
                                 + 1.09729 w_k1_INTC_s100
                                 + 1.38174 w_k1_KO_s100 + 1.27628 w_k1_MAT_s100
                                 + 1.09565 w_k1_MSFT_s100
                                 + 1.15328 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.2055 w_k1_ADM_s101
                                 + 1.24533 w_k1_AXP_s101 + 1.0281 w_k1_DIS_s101
                                 + 0.970437 w_k1_FDX_s101
                                 + 1.39274 w_k1_IBM_s101
                                 + 0.95632 w_k1_INTC_s101
                                 + 1.19217 w_k1_KO_s101 + 1.57859 w_k1_MAT_s101
                                 + 0.920969 w_k1_MSFT_s101
                                 + 1.12652 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14416 w_k1_ADM_s101
                                 + 1.12422 w_k1_AXP_s101
                                 + 1.22366 w_k1_DIS_s101 + 1.0998 w_k1_FDX_s101
                                 + 1.20303 w_k1_IBM_s101
                                 + 1.01916 w_k1_INTC_s101
                                 + 1.13156 w_k1_KO_s101 + 1.47348 w_k1_MAT_s101
                                 + 1.19481 w_k1_MSFT_s101
                                 + 1.16679 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.2578 w_k1_ADM_s101
                                 + 1.45845 w_k1_AXP_s101
                                 + 1.50086 w_k1_DIS_s101 + 1.236 w_k1_FDX_s101
                                 + 1.01287 w_k1_IBM_s101
                                 + 0.986505 w_k1_INTC_s101
                                 + 1.33024 w_k1_KO_s101 + 1.58548 w_k1_MAT_s101
                                 + 1.14526 w_k1_MSFT_s101
                                 + 1.20436 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.924196 w_k1_ADM_s101
                                 + 0.864892 w_k1_AXP_s101
                                 + 0.905728 w_k1_DIS_s101
                                 + 0.935345 w_k1_FDX_s101
                                 + 0.693286 w_k1_IBM_s101
                                 + 0.863427 w_k1_INTC_s101
                                 + 0.794176 w_k1_KO_s101
                                 + 0.947937 w_k1_MAT_s101
                                 + 0.919852 w_k1_MSFT_s101
                                 + 0.991602 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.858035 w_k1_ADM_s101
                                 + 0.692381 w_k1_AXP_s101
                                 + 0.676253 w_k1_DIS_s101
                                 + 0.713863 w_k1_FDX_s101
                                 + 1.00667 w_k1_IBM_s101
                                 + 0.743941 w_k1_INTC_s101
                                 + 0.744466 w_k1_KO_s101
                                 + 0.909869 w_k1_MAT_s101
                                 + 0.776085 w_k1_MSFT_s101
                                 + 0.890615 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.45175 w_k1_ADM_s101
                                 + 1.31908 w_k1_AXP_s101
                                 + 1.26953 w_k1_DIS_s101
                                 + 1.36379 w_k1_FDX_s101
                                 + 0.866219 w_k1_IBM_s101
                                 + 1.29364 w_k1_INTC_s101
                                 + 1.04501 w_k1_KO_s101
                                 + 0.736361 w_k1_MAT_s101
                                 + 1.15064 w_k1_MSFT_s101
                                 + 1.09118 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.35814 w_k1_ADM_s101
                                 + 1.72551 w_k1_AXP_s101
                                 + 1.75096 w_k1_DIS_s101
                                 + 1.45573 w_k1_FDX_s101
                                 + 1.32565 w_k1_IBM_s101
                                 + 1.12728 w_k1_INTC_s101
                                 + 1.27826 w_k1_KO_s101 + 2.68591 w_k1_MAT_s101
                                 + 1.25744 w_k1_MSFT_s101
                                 + 1.27044 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.05152 w_k1_ADM_s101
                                 + 1.14368 w_k1_AXP_s101
                                 + 0.993003 w_k1_DIS_s101
                                 + 0.955388 w_k1_FDX_s101
                                 + 0.969855 w_k1_IBM_s101
                                 + 0.947145 w_k1_INTC_s101
                                 + 1.17838 w_k1_KO_s101 + 1.43035 w_k1_MAT_s101
                                 + 0.946852 w_k1_MSFT_s101
                                 + 0.972202 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.954046 w_k1_ADM_s102
                                 + 1.0017 w_k1_AXP_s102 + 1.17207 w_k1_DIS_s102
                                 + 1.00455 w_k1_FDX_s102
                                 + 1.33726 w_k1_IBM_s102
                                 + 1.16839 w_k1_INTC_s102
                                 + 1.19416 w_k1_KO_s102 + 1.34125 w_k1_MAT_s102
                                 + 1.08774 w_k1_MSFT_s102
                                 + 1.08094 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.991049 w_k1_ADM_s102
                                 + 1.25631 w_k1_AXP_s102
                                 + 1.68149 w_k1_DIS_s102
                                 + 1.24203 w_k1_FDX_s102
                                 + 1.70234 w_k1_IBM_s102
                                 + 1.75168 w_k1_INTC_s102
                                 + 1.50555 w_k1_KO_s102 + 1.41281 w_k1_MAT_s102
                                 + 1.30068 w_k1_MSFT_s102
                                 + 1.27073 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10358 w_k1_ADM_s102
                                 + 1.28722 w_k1_AXP_s102
                                 + 1.03401 w_k1_DIS_s102
                                 + 1.06022 w_k1_FDX_s102
                                 + 0.988185 w_k1_IBM_s102
                                 + 1.1342 w_k1_INTC_s102 + 1.04775 w_k1_KO_s102
                                 + 1.23666 w_k1_MAT_s102
                                 + 0.881704 w_k1_MSFT_s102
                                 + 1.1269 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.39148 w_k1_ADM_s102
                                 + 1.52514 w_k1_AXP_s102
                                 + 1.49418 w_k1_DIS_s102
                                 + 1.39333 w_k1_FDX_s102
                                 + 1.10858 w_k1_IBM_s102
                                 + 1.50282 w_k1_INTC_s102
                                 + 1.22503 w_k1_KO_s102 + 1.06072 w_k1_MAT_s102
                                 + 1.21453 w_k1_MSFT_s102
                                 + 1.30469 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.02605 w_k1_ADM_s102
                                 + 1.1539 w_k1_AXP_s102 + 1.11648 w_k1_DIS_s102
                                 + 1.14428 w_k1_FDX_s102
                                 + 0.812177 w_k1_IBM_s102
                                 + 0.997991 w_k1_INTC_s102
                                 + 0.985464 w_k1_KO_s102
                                 + 0.83794 w_k1_MAT_s102
                                 + 1.03258 w_k1_MSFT_s102 + 1.0435 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.790975 w_k1_ADM_s102
                                 + 0.760994 w_k1_AXP_s102
                                 + 0.738556 w_k1_DIS_s102
                                 + 0.803654 w_k1_FDX_s102
                                 + 0.780975 w_k1_IBM_s102
                                 + 0.970388 w_k1_INTC_s102
                                 + 0.800889 w_k1_KO_s102
                                 + 0.757977 w_k1_MAT_s102
                                 + 0.724175 w_k1_MSFT_s102
                                 + 0.814276 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.836376 w_k1_ADM_s102
                                 + 0.826766 w_k1_AXP_s102
                                 + 0.947338 w_k1_DIS_s102
                                 + 0.936664 w_k1_FDX_s102
                                 + 0.905876 w_k1_IBM_s102
                                 + 1.0224 w_k1_INTC_s102
                                 + 0.913766 w_k1_KO_s102
                                 + 0.726175 w_k1_MAT_s102
                                 + 0.900391 w_k1_MSFT_s102
                                 + 0.898948 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.909676 w_k1_ADM_s102
                                 + 0.843637 w_k1_AXP_s102
                                 + 0.904558 w_k1_DIS_s102
                                 + 0.848004 w_k1_FDX_s102
                                 + 1.27588 w_k1_IBM_s102
                                 + 1.01611 w_k1_INTC_s102
                                 + 1.11136 w_k1_KO_s102 + 1.37822 w_k1_MAT_s102
                                 + 0.872683 w_k1_MSFT_s102
                                 + 1.11159 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.21405 w_k1_ADM_s103
                                 + 1.07661 w_k1_AXP_s103
                                 + 0.884184 w_k1_DIS_s103
                                 + 0.907996 w_k1_FDX_s103
                                 + 1.14323 w_k1_IBM_s103
                                 + 0.772085 w_k1_INTC_s103
                                 + 0.950036 w_k1_KO_s103
                                 + 1.16394 w_k1_MAT_s103
                                 + 0.850894 w_k1_MSFT_s103
                                 + 0.984042 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.823377 w_k1_ADM_s103
                                 + 0.55231 w_k1_AXP_s103
                                 + 0.574132 w_k1_DIS_s103
                                 + 0.727874 w_k1_FDX_s103
                                 + 0.626721 w_k1_IBM_s103
                                 + 0.797104 w_k1_INTC_s103
                                 + 0.538486 w_k1_KO_s103
                                 + 0.551731 w_k1_MAT_s103
                                 + 0.77804 w_k1_MSFT_s103
                                 + 0.749187 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.863638 w_k1_ADM_s103
                                 + 0.796194 w_k1_AXP_s103
                                 + 0.641083 w_k1_DIS_s103
                                 + 0.767442 w_k1_FDX_s103
                                 + 0.822675 w_k1_IBM_s103
                                 + 0.777451 w_k1_INTC_s103
                                 + 0.726586 w_k1_KO_s103
                                 + 0.882839 w_k1_MAT_s103
                                 + 0.734924 w_k1_MSFT_s103
                                 + 0.88112 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.25001 w_k1_ADM_s103
                                 + 1.47109 w_k1_AXP_s103
                                 + 1.45514 w_k1_DIS_s103
                                 + 1.27212 w_k1_FDX_s103
                                 + 1.37478 w_k1_IBM_s103
                                 + 1.07548 w_k1_INTC_s103
                                 + 1.19919 w_k1_KO_s103 + 1.98574 w_k1_MAT_s103
                                 + 1.24108 w_k1_MSFT_s103
                                 + 1.27367 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.833352 w_k1_ADM_s103
                                 + 0.812894 w_k1_AXP_s103
                                 + 0.798748 w_k1_DIS_s103
                                 + 0.723434 w_k1_FDX_s103
                                 + 1.24018 w_k1_IBM_s103
                                 + 0.773095 w_k1_INTC_s103
                                 + 0.862012 w_k1_KO_s103
                                 + 1.36745 w_k1_MAT_s103
                                 + 0.789494 w_k1_MSFT_s103
                                 + 0.993929 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.17651 w_k1_ADM_s103
                                 + 0.957718 w_k1_AXP_s103
                                 + 0.977213 w_k1_DIS_s103
                                 + 1.04195 w_k1_FDX_s103
                                 + 0.81915 w_k1_IBM_s103
                                 + 0.938224 w_k1_INTC_s103
                                 + 0.791686 w_k1_KO_s103
                                 + 1.03957 w_k1_MAT_s103
                                 + 0.93723 w_k1_MSFT_s103
                                 + 0.931094 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.37417 w_k1_ADM_s103
                                 + 1.24575 w_k1_AXP_s103 + 1.2666 w_k1_DIS_s103
                                 + 1.31495 w_k1_FDX_s103
                                 + 0.703194 w_k1_IBM_s103
                                 + 1.08708 w_k1_INTC_s103
                                 + 0.88344 w_k1_KO_s103 + 0.74515 w_k1_MAT_s103
                                 + 1.22839 w_k1_MSFT_s103
                                 + 1.01755 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.18171 w_k1_ADM_s103
                                 + 0.69765 w_k1_AXP_s103
                                 + 0.779053 w_k1_DIS_s103
                                 + 0.892501 w_k1_FDX_s103
                                 + 0.623107 w_k1_IBM_s103
                                 + 0.963841 w_k1_INTC_s103
                                 + 0.665768 w_k1_KO_s103
                                 + 0.703597 w_k1_MAT_s103
                                 + 0.900716 w_k1_MSFT_s103
                                 + 0.79835 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.772056 w_k1_ADM_s104
                                 + 0.780054 w_k1_AXP_s104 + 0.566 w_k1_DIS_s104
                                 + 0.661997 w_k1_FDX_s104
                                 + 0.800699 w_k1_IBM_s104
                                 + 0.869079 w_k1_INTC_s104
                                 + 0.75674 w_k1_KO_s104
                                 + 0.778513 w_k1_MAT_s104
                                 + 0.707963 w_k1_MSFT_s104
                                 + 0.928606 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.902737 w_k1_ADM_s104
                                 + 1.01485 w_k1_AXP_s104
                                 + 0.989362 w_k1_DIS_s104
                                 + 1.04936 w_k1_FDX_s104
                                 + 1.19966 w_k1_IBM_s104
                                 + 1.22926 w_k1_INTC_s104
                                 + 1.16402 w_k1_KO_s104 + 1.20458 w_k1_MAT_s104
                                 + 0.957358 w_k1_MSFT_s104
                                 + 1.08215 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.983116 w_k1_ADM_s104
                                 + 0.636745 w_k1_AXP_s104
                                 + 0.720796 w_k1_DIS_s104
                                 + 0.838446 w_k1_FDX_s104
                                 + 0.81113 w_k1_IBM_s104
                                 + 0.920648 w_k1_INTC_s104
                                 + 0.848817 w_k1_KO_s104
                                 + 0.77515 w_k1_MAT_s104
                                 + 0.861207 w_k1_MSFT_s104
                                 + 0.952731 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.813911 w_k1_ADM_s104
                                 + 0.984312 w_k1_AXP_s104
                                 + 0.751719 w_k1_DIS_s104
                                 + 0.955572 w_k1_FDX_s104
                                 + 0.791078 w_k1_IBM_s104
                                 + 1.25272 w_k1_INTC_s104
                                 + 0.823379 w_k1_KO_s104
                                 + 0.692817 w_k1_MAT_s104
                                 + 0.926829 w_k1_MSFT_s104
                                 + 0.966093 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.25532 w_k1_ADM_s104
                                 + 1.11916 w_k1_AXP_s104
                                 + 1.12915 w_k1_DIS_s104
                                 + 1.16651 w_k1_FDX_s104
                                 + 0.751354 w_k1_IBM_s104
                                 + 1.49143 w_k1_INTC_s104
                                 + 0.957093 w_k1_KO_s104
                                 + 0.844269 w_k1_MAT_s104
                                 + 1.08095 w_k1_MSFT_s104
                                 + 1.05534 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.06217 w_k1_ADM_s104
                                 + 1.3581 w_k1_AXP_s104 + 1.37733 w_k1_DIS_s104
                                 + 1.22199 w_k1_FDX_s104 + 1.3304 w_k1_IBM_s104
                                 + 1.29457 w_k1_INTC_s104
                                 + 1.29128 w_k1_KO_s104 + 1.8356 w_k1_MAT_s104
                                 + 1.26423 w_k1_MSFT_s104
                                 + 1.31195 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.911237 w_k1_ADM_s104
                                 + 0.881276 w_k1_AXP_s104
                                 + 1.23826 w_k1_DIS_s104 + 1.0663 w_k1_FDX_s104
                                 + 1.10131 w_k1_IBM_s104
                                 + 1.30552 w_k1_INTC_s104
                                 + 1.11988 w_k1_KO_s104
                                 + 0.798868 w_k1_MAT_s104
                                 + 1.25565 w_k1_MSFT_s104 + 1.1423 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.911541 w_k1_ADM_s104
                                 + 0.860891 w_k1_AXP_s104
                                 + 0.843388 w_k1_DIS_s104
                                 + 0.906998 w_k1_FDX_s104
                                 + 1.09789 w_k1_IBM_s104
                                 + 0.894137 w_k1_INTC_s104
                                 + 0.953844 w_k1_KO_s104
                                 + 1.01708 w_k1_MAT_s104
                                 + 0.808695 w_k1_MSFT_s104
                                 + 1.0192 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.5444 w_k1_ADM_s105
                                 + 1.19336 w_k1_AXP_s105
                                 + 1.55136 w_k1_DIS_s105
                                 + 1.16334 w_k1_FDX_s105
                                 + 1.88052 w_k1_IBM_s105
                                 + 0.976863 w_k1_INTC_s105
                                 + 1.52051 w_k1_KO_s105 + 1.96274 w_k1_MAT_s105
                                 + 0.907379 w_k1_MSFT_s105
                                 + 1.23802 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.733714 w_k1_ADM_s105
                                 + 0.527651 w_k1_AXP_s105
                                 + 0.590864 w_k1_DIS_s105
                                 + 0.700323 w_k1_FDX_s105
                                 + 0.693829 w_k1_IBM_s105
                                 + 0.701577 w_k1_INTC_s105
                                 + 0.670226 w_k1_KO_s105
                                 + 0.594307 w_k1_MAT_s105
                                 + 0.620679 w_k1_MSFT_s105
                                 + 0.781482 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.09947 w_k1_ADM_s105
                                 + 0.947688 w_k1_AXP_s105
                                 + 0.822823 w_k1_DIS_s105
                                 + 1.04229 w_k1_FDX_s105
                                 + 0.731047 w_k1_IBM_s105
                                 + 0.946838 w_k1_INTC_s105
                                 + 0.867504 w_k1_KO_s105
                                 + 0.856583 w_k1_MAT_s105
                                 + 0.946814 w_k1_MSFT_s105
                                 + 0.942167 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.951702 w_k1_ADM_s105
                                 + 0.654763 w_k1_AXP_s105
                                 + 0.660396 w_k1_DIS_s105
                                 + 0.707749 w_k1_FDX_s105
                                 + 1.01335 w_k1_IBM_s105
                                 + 0.738019 w_k1_INTC_s105
                                 + 0.780963 w_k1_KO_s105
                                 + 0.825835 w_k1_MAT_s105
                                 + 0.675821 w_k1_MSFT_s105
                                 + 0.746825 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.23694 w_k1_ADM_s105
                                 + 1.41607 w_k1_AXP_s105
                                 + 1.71589 w_k1_DIS_s105
                                 + 1.41179 w_k1_FDX_s105
                                 + 1.17349 w_k1_IBM_s105
                                 + 1.11667 w_k1_INTC_s105
                                 + 1.31669 w_k1_KO_s105 + 1.4613 w_k1_MAT_s105
                                 + 1.32442 w_k1_MSFT_s105
                                 + 1.19436 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.14488 w_k1_ADM_s105
                                 + 1.02553 w_k1_AXP_s105
                                 + 1.16852 w_k1_DIS_s105
                                 + 1.10461 w_k1_FDX_s105
                                 + 0.961415 w_k1_IBM_s105
                                 + 0.986854 w_k1_INTC_s105
                                 + 1.01839 w_k1_KO_s105 + 1.08039 w_k1_MAT_s105
                                 + 0.979313 w_k1_MSFT_s105
                                 + 0.913446 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.820697 w_k1_ADM_s105
                                 + 0.591143 w_k1_AXP_s105
                                 + 1.00732 w_k1_DIS_s105
                                 + 0.893488 w_k1_FDX_s105
                                 + 0.750992 w_k1_IBM_s105
                                 + 0.997293 w_k1_INTC_s105
                                 + 0.958333 w_k1_KO_s105
                                 + 0.536576 w_k1_MAT_s105
                                 + 0.946302 w_k1_MSFT_s105
                                 + 0.873702 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.86587 w_k1_ADM_s105
                                 + 0.984522 w_k1_AXP_s105
                                 + 0.912657 w_k1_DIS_s105
                                 + 0.902797 w_k1_FDX_s105
                                 + 1.27095 w_k1_IBM_s105
                                 + 1.02171 w_k1_INTC_s105 + 1.1281 w_k1_KO_s105
                                 + 1.26184 w_k1_MAT_s105
                                 + 0.909923 w_k1_MSFT_s105
                                 + 0.983397 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.9015 w_k1_ADM_s106
                                 + 1.01173 w_k1_AXP_s106
                                 + 0.982011 w_k1_DIS_s106
                                 + 0.92759 w_k1_FDX_s106
                                 + 1.41658 w_k1_IBM_s106
                                 + 1.16474 w_k1_INTC_s106
                                 + 0.999422 w_k1_KO_s106
                                 + 1.11393 w_k1_MAT_s106
                                 + 0.900974 w_k1_MSFT_s106
                                 + 0.9678 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12006 w_k1_ADM_s106
                                 + 1.02949 w_k1_AXP_s106
                                 + 0.904868 w_k1_DIS_s106
                                 + 1.11858 w_k1_FDX_s106
                                 + 0.665349 w_k1_IBM_s106
                                 + 1.15028 w_k1_INTC_s106
                                 + 0.868724 w_k1_KO_s106
                                 + 0.782788 w_k1_MAT_s106
                                 + 1.04184 w_k1_MSFT_s106
                                 + 0.947698 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.973819 w_k1_ADM_s106
                                 + 1.046 w_k1_AXP_s106 + 0.801232 w_k1_DIS_s106
                                 + 0.909958 w_k1_FDX_s106
                                 + 1.0764 w_k1_IBM_s106 + 1.0437 w_k1_INTC_s106
                                 + 0.988698 w_k1_KO_s106
                                 + 1.09397 w_k1_MAT_s106
                                 + 0.825622 w_k1_MSFT_s106
                                 + 1.02878 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.17064 w_k1_ADM_s106
                                 + 1.43915 w_k1_AXP_s106
                                 + 1.51708 w_k1_DIS_s106
                                 + 1.17967 w_k1_FDX_s106
                                 + 1.57545 w_k1_IBM_s106
                                 + 1.22593 w_k1_INTC_s106
                                 + 1.30339 w_k1_KO_s106 + 1.86819 w_k1_MAT_s106
                                 + 1.27705 w_k1_MSFT_s106
                                 + 1.21411 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.36146 w_k1_ADM_s106
                                 + 1.39059 w_k1_AXP_s106
                                 + 1.67349 w_k1_DIS_s106
                                 + 1.37078 w_k1_FDX_s106
                                 + 0.954722 w_k1_IBM_s106
                                 + 1.53981 w_k1_INTC_s106
                                 + 1.34888 w_k1_KO_s106 + 1.05997 w_k1_MAT_s106
                                 + 1.49497 w_k1_MSFT_s106
                                 + 1.11181 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.87201 w_k1_ADM_s106
                                 + 0.82076 w_k1_AXP_s106
                                 + 0.797554 w_k1_DIS_s106
                                 + 0.88387 w_k1_FDX_s106
                                 + 0.849345 w_k1_IBM_s106
                                 + 1.06729 w_k1_INTC_s106
                                 + 0.868546 w_k1_KO_s106
                                 + 0.749512 w_k1_MAT_s106
                                 + 0.911117 w_k1_MSFT_s106
                                 + 0.849644 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.24129 w_k1_ADM_s106
                                 + 1.18559 w_k1_AXP_s106 + 1.1319 w_k1_DIS_s106
                                 + 1.10801 w_k1_FDX_s106
                                 + 1.02605 w_k1_IBM_s106
                                 + 1.27858 w_k1_INTC_s106
                                 + 1.19841 w_k1_KO_s106
                                 + 0.777186 w_k1_MAT_s106
                                 + 1.09251 w_k1_MSFT_s106
                                 + 1.07929 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.719518 w_k1_ADM_s106
                                 + 0.647247 w_k1_AXP_s106
                                 + 0.505361 w_k1_DIS_s106
                                 + 0.70886 w_k1_FDX_s106
                                 + 0.768908 w_k1_IBM_s106
                                 + 0.909685 w_k1_INTC_s106
                                 + 0.70082 w_k1_KO_s106
                                 + 0.675991 w_k1_MAT_s106
                                 + 0.779194 w_k1_MSFT_s106
                                 + 0.795658 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08952 w_k1_ADM_s107
                                 + 0.722109 w_k1_AXP_s107
                                 + 0.537715 w_k1_DIS_s107
                                 + 0.799882 w_k1_FDX_s107
                                 + 0.705597 w_k1_IBM_s107
                                 + 0.818282 w_k1_INTC_s107
                                 + 0.701891 w_k1_KO_s107
                                 + 0.648832 w_k1_MAT_s107
                                 + 0.834673 w_k1_MSFT_s107
                                 + 0.837673 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.24258 w_k1_ADM_s107
                                 + 1.18527 w_k1_AXP_s107
                                 + 1.26984 w_k1_DIS_s107
                                 + 1.29413 w_k1_FDX_s107 + 0.7674 w_k1_IBM_s107
                                 + 1.21863 w_k1_INTC_s107
                                 + 1.22518 w_k1_KO_s107
                                 + 0.734239 w_k1_MAT_s107
                                 + 1.2969 w_k1_MSFT_s107 + 1.06955 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.19498 w_k1_ADM_s107
                                 + 1.36953 w_k1_AXP_s107
                                 + 1.70355 w_k1_DIS_s107
                                 + 1.24486 w_k1_FDX_s107
                                 + 1.64167 w_k1_IBM_s107
                                 + 1.14104 w_k1_INTC_s107
                                 + 1.76153 w_k1_KO_s107 + 1.73814 w_k1_MAT_s107
                                 + 1.21508 w_k1_MSFT_s107
                                 + 1.15617 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.29593 w_k1_ADM_s107
                                 + 1.04153 w_k1_AXP_s107
                                 + 0.895031 w_k1_DIS_s107
                                 + 1.04107 w_k1_FDX_s107
                                 + 0.938414 w_k1_IBM_s107
                                 + 0.859076 w_k1_INTC_s107
                                 + 0.978444 w_k1_KO_s107
                                 + 1.16303 w_k1_MAT_s107
                                 + 0.951311 w_k1_MSFT_s107
                                 + 1.00155 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.19459 w_k1_ADM_s107
                                 + 1.12368 w_k1_AXP_s107
                                 + 1.03772 w_k1_DIS_s107
                                 + 1.17196 w_k1_FDX_s107
                                 + 1.04654 w_k1_IBM_s107 + 1.113 w_k1_INTC_s107
                                 + 1.21623 w_k1_KO_s107 + 1.01794 w_k1_MAT_s107
                                 + 1.15352 w_k1_MSFT_s107
                                 + 1.13493 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.02313 w_k1_ADM_s107
                                 + 1.17227 w_k1_AXP_s107
                                 + 0.806978 w_k1_DIS_s107
                                 + 0.939494 w_k1_FDX_s107
                                 + 1.33363 w_k1_IBM_s107
                                 + 0.893853 w_k1_INTC_s107
                                 + 1.09923 w_k1_KO_s107 + 1.32233 w_k1_MAT_s107
                                 + 0.83878 w_k1_MSFT_s107
                                 + 0.965841 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.4386 w_k1_ADM_s107
                                 + 1.34416 w_k1_AXP_s107 + 1.6432 w_k1_DIS_s107
                                 + 1.44473 w_k1_FDX_s107 + 1.223 w_k1_IBM_s107
                                 + 1.25748 w_k1_INTC_s107
                                 + 1.32931 w_k1_KO_s107 + 1.17997 w_k1_MAT_s107
                                 + 1.47972 w_k1_MSFT_s107
                                 + 1.31434 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.1166 w_k1_ADM_s107
                                 + 0.820836 w_k1_AXP_s107
                                 + 0.965961 w_k1_DIS_s107
                                 + 0.931211 w_k1_FDX_s107
                                 + 0.871015 w_k1_IBM_s107
                                 + 1.03037 w_k1_INTC_s107
                                 + 1.07126 w_k1_KO_s107
                                 + 0.719779 w_k1_MAT_s107
                                 + 1.08946 w_k1_MSFT_s107
                                 + 0.933284 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.11928 w_k1_ADM_s110
                                 + 1.32455 w_k1_AXP_s110
                                 + 1.19857 w_k1_DIS_s110
                                 + 1.23699 w_k1_FDX_s110
                                 + 0.81134 w_k1_IBM_s110
                                 + 1.16329 w_k1_INTC_s110
                                 + 1.15607 w_k1_KO_s110 + 1.58714 w_k1_MAT_s110
                                 + 1.28203 w_k1_MSFT_s110
                                 + 1.24306 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.809776 w_k1_ADM_s110
                                 + 0.799177 w_k1_AXP_s110
                                 + 0.816754 w_k1_DIS_s110
                                 + 0.874118 w_k1_FDX_s110
                                 + 1.08554 w_k1_IBM_s110
                                 + 1.0193 w_k1_INTC_s110 + 1.03783 w_k1_KO_s110
                                 + 1.66023 w_k1_MAT_s110
                                 + 0.893292 w_k1_MSFT_s110
                                 + 0.95413 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.992405 w_k1_ADM_s110
                                 + 0.82482 w_k1_AXP_s110
                                 + 0.947827 w_k1_DIS_s110
                                 + 1.02072 w_k1_FDX_s110
                                 + 0.850328 w_k1_IBM_s110
                                 + 1.07253 w_k1_INTC_s110
                                 + 0.92692 w_k1_KO_s110
                                 + 0.929793 w_k1_MAT_s110
                                 + 1.07832 w_k1_MSFT_s110
                                 + 0.91792 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.06747 w_k1_ADM_s110
                                 + 1.17686 w_k1_AXP_s110
                                 + 1.09682 w_k1_DIS_s110
                                 + 1.11314 w_k1_FDX_s110
                                 + 0.963643 w_k1_IBM_s110
                                 + 0.787534 w_k1_INTC_s110
                                 + 1.04412 w_k1_KO_s110 + 1.73762 w_k1_MAT_s110
                                 + 1.00409 w_k1_MSFT_s110
                                 + 1.07936 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.906455 w_k1_ADM_s110
                                 + 0.781119 w_k1_AXP_s110
                                 + 1.02024 w_k1_DIS_s110
                                 + 0.960181 w_k1_FDX_s110
                                 + 1.31858 w_k1_IBM_s110
                                 + 0.917489 w_k1_INTC_s110
                                 + 1.23623 w_k1_KO_s110 + 1.2363 w_k1_MAT_s110
                                 + 0.97159 w_k1_MSFT_s110
                                 + 1.00842 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.782026 w_k1_ADM_s110
                                 + 1.41059 w_k1_AXP_s110
                                 + 1.43473 w_k1_DIS_s110
                                 + 1.22849 w_k1_FDX_s110
                                 + 1.44944 w_k1_IBM_s110
                                 + 1.07388 w_k1_INTC_s110
                                 + 1.36584 w_k1_KO_s110 + 1.83296 w_k1_MAT_s110
                                 + 1.1183 w_k1_MSFT_s110 + 1.17229 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.675502 w_k1_ADM_s110
                                 + 0.605105 w_k1_AXP_s110
                                 + 0.641455 w_k1_DIS_s110
                                 + 0.817107 w_k1_FDX_s110
                                 + 0.590804 w_k1_IBM_s110
                                 + 0.93346 w_k1_INTC_s110
                                 + 0.824413 w_k1_KO_s110
                                 + 0.780442 w_k1_MAT_s110
                                 + 0.981372 w_k1_MSFT_s110
                                 + 0.847751 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.800656 w_k1_ADM_s110
                                 + 0.702814 w_k1_AXP_s110
                                 + 0.691724 w_k1_DIS_s110
                                 + 0.80187 w_k1_FDX_s110
                                 + 0.877369 w_k1_IBM_s110
                                 + 0.875344 w_k1_INTC_s110
                                 + 0.84469 w_k1_KO_s110
                                 + 0.802063 w_k1_MAT_s110
                                 + 0.880506 w_k1_MSFT_s110
                                 + 0.844993 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.933594 w_k1_ADM_s111
                                 + 1.059 w_k1_AXP_s111 + 1.11543 w_k1_DIS_s111
                                 + 1.03398 w_k1_FDX_s111
                                 + 1.26435 w_k1_IBM_s111
                                 + 1.08757 w_k1_INTC_s111
                                 + 1.12802 w_k1_KO_s111 + 1.05348 w_k1_MAT_s111
                                 + 0.946833 w_k1_MSFT_s111
                                 + 0.985858 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01471 w_k1_ADM_s111
                                 + 1.2428 w_k1_AXP_s111 + 1.00827 w_k1_DIS_s111
                                 + 1.11721 w_k1_FDX_s111
                                 + 0.821087 w_k1_IBM_s111
                                 + 0.941823 w_k1_INTC_s111
                                 + 0.817368 w_k1_KO_s111
                                 + 1.19785 w_k1_MAT_s111
                                 + 0.969175 w_k1_MSFT_s111
                                 + 1.03481 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.00679 w_k1_ADM_s111
                                 + 0.88242 w_k1_AXP_s111
                                 + 1.00166 w_k1_DIS_s111
                                 + 1.12432 w_k1_FDX_s111
                                 + 0.736475 w_k1_IBM_s111
                                 + 1.01846 w_k1_INTC_s111
                                 + 0.656628 w_k1_KO_s111
                                 + 0.560781 w_k1_MAT_s111
                                 + 0.987787 w_k1_MSFT_s111
                                 + 0.896236 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.762046 w_k1_ADM_s111
                                 + 0.771454 w_k1_AXP_s111
                                 + 0.792643 w_k1_DIS_s111
                                 + 0.944154 w_k1_FDX_s111
                                 + 0.750581 w_k1_IBM_s111
                                 + 0.930325 w_k1_INTC_s111
                                 + 0.772178 w_k1_KO_s111
                                 + 0.548781 w_k1_MAT_s111
                                 + 0.849511 w_k1_MSFT_s111
                                 + 0.849283 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.05968 w_k1_ADM_s111
                                 + 1.05579 w_k1_AXP_s111
                                 + 1.10808 w_k1_DIS_s111
                                 + 1.09163 w_k1_FDX_s111
                                 + 1.05184 w_k1_IBM_s111
                                 + 1.06871 w_k1_INTC_s111
                                 + 0.972885 w_k1_KO_s111
                                 + 1.04462 w_k1_MAT_s111
                                 + 0.95006 w_k1_MSFT_s111
                                 + 0.978645 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.09093 w_k1_ADM_s111
                                 + 1.26852 w_k1_AXP_s111
                                 + 1.35589 w_k1_DIS_s111
                                 + 1.06602 w_k1_FDX_s111
                                 + 1.61952 w_k1_IBM_s111
                                 + 1.21476 w_k1_INTC_s111
                                 + 1.46153 w_k1_KO_s111 + 1.33562 w_k1_MAT_s111
                                 + 1.05387 w_k1_MSFT_s111
                                 + 1.19326 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.18895 w_k1_ADM_s111
                                 + 1.33519 w_k1_AXP_s111
                                 + 1.75211 w_k1_DIS_s111
                                 + 1.53924 w_k1_FDX_s111
                                 + 1.19549 w_k1_IBM_s111
                                 + 1.34241 w_k1_INTC_s111
                                 + 1.26586 w_k1_KO_s111 + 1.12754 w_k1_MAT_s111
                                 + 1.45716 w_k1_MSFT_s111
                                 + 1.15181 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.673645 w_k1_ADM_s111
                                 + 0.772655 w_k1_AXP_s111
                                 + 0.743303 w_k1_DIS_s111
                                 + 0.765032 w_k1_FDX_s111
                                 + 0.941756 w_k1_IBM_s111
                                 + 0.817332 w_k1_INTC_s111
                                 + 0.748241 w_k1_KO_s111
                                 + 0.939557 w_k1_MAT_s111
                                 + 0.74936 w_k1_MSFT_s111
                                 + 0.907598 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.782573 w_k1_ADM_s112
                                 + 0.761064 w_k1_AXP_s112
                                 + 0.618495 w_k1_DIS_s112
                                 + 0.761597 w_k1_FDX_s112
                                 + 0.764818 w_k1_IBM_s112
                                 + 0.749116 w_k1_INTC_s112
                                 + 0.798771 w_k1_KO_s112
                                 + 0.882254 w_k1_MAT_s112
                                 + 0.830134 w_k1_MSFT_s112
                                 + 0.847693 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.747108 w_k1_ADM_s112
                                 + 0.799089 w_k1_AXP_s112
                                 + 0.665289 w_k1_DIS_s112
                                 + 0.78011 w_k1_FDX_s112
                                 + 1.04612 w_k1_IBM_s112
                                 + 0.887008 w_k1_INTC_s112
                                 + 0.778211 w_k1_KO_s112
                                 + 0.670768 w_k1_MAT_s112
                                 + 0.76607 w_k1_MSFT_s112
                                 + 0.895803 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.972009 w_k1_ADM_s112
                                 + 1.27028 w_k1_AXP_s112
                                 + 1.58233 w_k1_DIS_s112
                                 + 1.34219 w_k1_FDX_s112
                                 + 1.09422 w_k1_IBM_s112
                                 + 1.40584 w_k1_INTC_s112
                                 + 1.22068 w_k1_KO_s112
                                 + 0.994652 w_k1_MAT_s112
                                 + 1.34057 w_k1_MSFT_s112
                                 + 1.35171 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.931925 w_k1_ADM_s112
                                 + 0.779983 w_k1_AXP_s112
                                 + 0.874667 w_k1_DIS_s112
                                 + 0.867269 w_k1_FDX_s112
                                 + 0.96283 w_k1_IBM_s112
                                 + 0.764204 w_k1_INTC_s112
                                 + 1.06556 w_k1_KO_s112
                                 + 0.651416 w_k1_MAT_s112
                                 + 0.938044 w_k1_MSFT_s112
                                 + 0.932047 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.96473 w_k1_ADM_s112
                                 + 1.18054 w_k1_AXP_s112 + 1.2261 w_k1_DIS_s112
                                 + 1.10697 w_k1_FDX_s112
                                 + 1.33621 w_k1_IBM_s112
                                 + 0.960976 w_k1_INTC_s112
                                 + 1.32608 w_k1_KO_s112 + 1.27774 w_k1_MAT_s112
                                 + 1.12934 w_k1_MSFT_s112
                                 + 1.18015 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.967339 w_k1_ADM_s112
                                 + 0.932964 w_k1_AXP_s112
                                 + 0.923631 w_k1_DIS_s112
                                 + 0.959721 w_k1_FDX_s112
                                 + 1.18568 w_k1_IBM_s112
                                 + 0.774633 w_k1_INTC_s112
                                 + 1.04564 w_k1_KO_s112 + 1.12166 w_k1_MAT_s112
                                 + 0.975333 w_k1_MSFT_s112
                                 + 1.05567 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.921438 w_k1_ADM_s112
                                 + 0.898543 w_k1_AXP_s112
                                 + 0.796971 w_k1_DIS_s112
                                 + 0.954276 w_k1_FDX_s112
                                 + 0.768832 w_k1_IBM_s112
                                 + 0.652088 w_k1_INTC_s112
                                 + 0.846313 w_k1_KO_s112
                                 + 0.543891 w_k1_MAT_s112
                                 + 0.854708 w_k1_MSFT_s112
                                 + 1.03484 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.00826 w_k1_ADM_s112
                                 + 1.11325 w_k1_AXP_s112 + 1.0474 w_k1_DIS_s112
                                 + 1.15252 w_k1_FDX_s112
                                 + 0.661291 w_k1_IBM_s112
                                 + 1.01441 w_k1_INTC_s112
                                 + 0.875897 w_k1_KO_s112
                                 + 0.59974 w_k1_MAT_s112
                                 + 1.04743 w_k1_MSFT_s112
                                 + 1.12362 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.866284 w_k1_ADM_s113
                                 + 0.662453 w_k1_AXP_s113
                                 + 0.660724 w_k1_DIS_s113
                                 + 0.847133 w_k1_FDX_s113
                                 + 0.566097 w_k1_IBM_s113
                                 + 1.05527 w_k1_INTC_s113
                                 + 0.621766 w_k1_KO_s113
                                 + 0.384038 w_k1_MAT_s113
                                 + 0.780256 w_k1_MSFT_s113
                                 + 0.804287 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.928188 w_k1_ADM_s113
                                 + 0.921153 w_k1_AXP_s113
                                 + 0.98243 w_k1_DIS_s113
                                 + 1.04125 w_k1_FDX_s113
                                 + 1.42396 w_k1_IBM_s113
                                 + 1.12138 w_k1_INTC_s113
                                 + 1.19235 w_k1_KO_s113 + 1.3405 w_k1_MAT_s113
                                 + 1.03017 w_k1_MSFT_s113
                                 + 1.01317 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.802556 w_k1_ADM_s113
                                 + 0.655809 w_k1_AXP_s113
                                 + 1.05171 w_k1_DIS_s113
                                 + 0.910732 w_k1_FDX_s113
                                 + 1.01913 w_k1_IBM_s113 + 1.12 w_k1_INTC_s113
                                 + 1.04957 w_k1_KO_s113 + 1.07603 w_k1_MAT_s113
                                 + 1.04654 w_k1_MSFT_s113
                                 + 0.95664 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.84961 w_k1_ADM_s113
                                 + 1.03735 w_k1_AXP_s113
                                 + 0.633668 w_k1_DIS_s113
                                 + 0.927313 w_k1_FDX_s113
                                 + 1.05834 w_k1_IBM_s113
                                 + 1.03025 w_k1_INTC_s113
                                 + 0.685118 w_k1_KO_s113
                                 + 0.904446 w_k1_MAT_s113
                                 + 0.755984 w_k1_MSFT_s113
                                 + 0.802923 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.19804 w_k1_ADM_s113
                                 + 1.44118 w_k1_AXP_s113
                                 + 1.57296 w_k1_DIS_s113
                                 + 1.30982 w_k1_FDX_s113
                                 + 1.50179 w_k1_IBM_s113
                                 + 1.58802 w_k1_INTC_s113
                                 + 1.41315 w_k1_KO_s113 + 1.8328 w_k1_MAT_s113
                                 + 1.51993 w_k1_MSFT_s113
                                 + 1.14696 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.06312 w_k1_ADM_s113
                                 + 0.899515 w_k1_AXP_s113
                                 + 0.880925 w_k1_DIS_s113
                                 + 0.998731 w_k1_FDX_s113
                                 + 0.838055 w_k1_IBM_s113
                                 + 1.0135 w_k1_INTC_s113
                                 + 0.735694 w_k1_KO_s113
                                 + 0.705887 w_k1_MAT_s113
                                 + 0.874512 w_k1_MSFT_s113
                                 + 0.860726 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.05869 w_k1_ADM_s113
                                 + 0.980424 w_k1_AXP_s113
                                 + 1.38352 w_k1_DIS_s113
                                 + 1.26993 w_k1_FDX_s113
                                 + 0.913359 w_k1_IBM_s113
                                 + 1.33906 w_k1_INTC_s113
                                 + 0.974244 w_k1_KO_s113
                                 + 0.904591 w_k1_MAT_s113
                                 + 1.18573 w_k1_MSFT_s113
                                 + 0.997236 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.773911 w_k1_ADM_s113
                                 + 0.563452 w_k1_AXP_s113
                                 + 0.540398 w_k1_DIS_s113
                                 + 0.779455 w_k1_FDX_s113
                                 + 0.487252 w_k1_IBM_s113
                                 + 0.718578 w_k1_INTC_s113
                                 + 0.530229 w_k1_KO_s113
                                 + 0.457636 w_k1_MAT_s113
                                 + 0.763593 w_k1_MSFT_s113
                                 + 0.648489 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.29218 w_k1_ADM_s114
                                 + 1.78175 w_k1_AXP_s114
                                 + 1.50622 w_k1_DIS_s114
                                 + 1.35656 w_k1_FDX_s114
                                 + 1.62605 w_k1_IBM_s114
                                 + 1.19422 w_k1_INTC_s114
                                 + 1.29792 w_k1_KO_s114 + 1.88425 w_k1_MAT_s114
                                 + 1.07701 w_k1_MSFT_s114
                                 + 1.25762 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.948087 w_k1_ADM_s114
                                 + 0.929804 w_k1_AXP_s114
                                 + 1.02595 w_k1_DIS_s114
                                 + 1.02775 w_k1_FDX_s114
                                 + 0.957549 w_k1_IBM_s114
                                 + 0.998116 w_k1_INTC_s114
                                 + 1.00822 w_k1_KO_s114
                                 + 0.756246 w_k1_MAT_s114
                                 + 0.983779 w_k1_MSFT_s114
                                 + 1.0492 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.871736 w_k1_ADM_s114
                                 + 0.944381 w_k1_AXP_s114
                                 + 0.960921 w_k1_DIS_s114
                                 + 0.925063 w_k1_FDX_s114
                                 + 1.34567 w_k1_IBM_s114
                                 + 0.830221 w_k1_INTC_s114
                                 + 0.990947 w_k1_KO_s114
                                 + 1.11516 w_k1_MAT_s114
                                 + 0.99401 w_k1_MSFT_s114
                                 + 1.08122 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.1115 w_k1_ADM_s114
                                 + 1.43897 w_k1_AXP_s114 + 1.0421 w_k1_DIS_s114
                                 + 1.23879 w_k1_FDX_s114
                                 + 0.780336 w_k1_IBM_s114
                                 + 1.03409 w_k1_INTC_s114
                                 + 0.900753 w_k1_KO_s114
                                 + 1.09464 w_k1_MAT_s114
                                 + 1.00927 w_k1_MSFT_s114
                                 + 1.20566 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.723403 w_k1_ADM_s114
                                 + 0.737898 w_k1_AXP_s114
                                 + 0.656946 w_k1_DIS_s114
                                 + 0.731743 w_k1_FDX_s114
                                 + 0.909462 w_k1_IBM_s114
                                 + 0.683494 w_k1_INTC_s114
                                 + 0.844668 w_k1_KO_s114
                                 + 0.988542 w_k1_MAT_s114
                                 + 0.787781 w_k1_MSFT_s114
                                 + 0.946141 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.11724 w_k1_ADM_s114
                                 + 1.37386 w_k1_AXP_s114
                                 + 1.12827 w_k1_DIS_s114
                                 + 1.22614 w_k1_FDX_s114
                                 + 1.13195 w_k1_IBM_s114
                                 + 1.16521 w_k1_INTC_s114
                                 + 1.02217 w_k1_KO_s114 + 1.18043 w_k1_MAT_s114
                                 + 1.01084 w_k1_MSFT_s114
                                 + 1.14954 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.8981 w_k1_ADM_s114
                                 + 0.804856 w_k1_AXP_s114
                                 + 0.779711 w_k1_DIS_s114
                                 + 0.91302 w_k1_FDX_s114
                                 + 0.635988 w_k1_IBM_s114
                                 + 0.763317 w_k1_INTC_s114
                                 + 0.700955 w_k1_KO_s114
                                 + 0.666754 w_k1_MAT_s114
                                 + 0.870436 w_k1_MSFT_s114
                                 + 0.884708 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.2771 w_k1_ADM_s114
                                 + 1.36921 w_k1_AXP_s114
                                 + 1.64185 w_k1_DIS_s114
                                 + 1.45327 w_k1_FDX_s114
                                 + 1.12334 w_k1_IBM_s114
                                 + 1.31465 w_k1_INTC_s114
                                 + 1.34737 w_k1_KO_s114 + 1.13 w_k1_MAT_s114
                                 + 1.32685 w_k1_MSFT_s114
                                 + 1.30395 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.783051 w_k1_ADM_s115
                                 + 0.863955 w_k1_AXP_s115
                                 + 0.815859 w_k1_DIS_s115
                                 + 0.968366 w_k1_FDX_s115
                                 + 0.839055 w_k1_IBM_s115
                                 + 0.938631 w_k1_INTC_s115
                                 + 0.789748 w_k1_KO_s115
                                 + 0.876844 w_k1_MAT_s115
                                 + 0.930317 w_k1_MSFT_s115
                                 + 0.914466 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.17646 w_k1_ADM_s115
                                 + 1.20752 w_k1_AXP_s115
                                 + 1.32882 w_k1_DIS_s115
                                 + 1.18452 w_k1_FDX_s115
                                 + 1.17741 w_k1_IBM_s115
                                 + 1.27155 w_k1_INTC_s115
                                 + 1.12552 w_k1_KO_s115 + 1.22008 w_k1_MAT_s115
                                 + 1.22982 w_k1_MSFT_s115
                                 + 1.20487 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.31079 w_k1_ADM_s115
                                 + 1.83196 w_k1_AXP_s115
                                 + 1.70683 w_k1_DIS_s115 + 1.569 w_k1_FDX_s115
                                 + 1.42062 w_k1_IBM_s115
                                 + 1.62553 w_k1_INTC_s115
                                 + 1.44346 w_k1_KO_s115 + 2.06282 w_k1_MAT_s115
                                 + 1.60752 w_k1_MSFT_s115
                                 + 1.32935 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.04563 w_k1_ADM_s115
                                 + 1.11559 w_k1_AXP_s115
                                 + 1.37735 w_k1_DIS_s115
                                 + 1.04782 w_k1_FDX_s115
                                 + 1.73451 w_k1_IBM_s115
                                 + 1.24647 w_k1_INTC_s115
                                 + 1.37422 w_k1_KO_s115 + 1.86121 w_k1_MAT_s115
                                 + 1.18631 w_k1_MSFT_s115 + 1.2155 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.950283 w_k1_ADM_s115
                                 + 0.731659 w_k1_AXP_s115
                                 + 0.634708 w_k1_DIS_s115
                                 + 0.80224 w_k1_FDX_s115
                                 + 0.630925 w_k1_IBM_s115
                                 + 0.872943 w_k1_INTC_s115
                                 + 0.712186 w_k1_KO_s115
                                 + 0.66261 w_k1_MAT_s115
                                 + 0.91039 w_k1_MSFT_s115
                                 + 0.846092 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.06514 w_k1_ADM_s115
                                 + 1.06711 w_k1_AXP_s115
                                 + 1.35489 w_k1_DIS_s115
                                 + 1.27722 w_k1_FDX_s115
                                 + 0.99656 w_k1_IBM_s115
                                 + 1.31643 w_k1_INTC_s115
                                 + 1.08372 w_k1_KO_s115
                                 + 0.834366 w_k1_MAT_s115
                                 + 1.45645 w_k1_MSFT_s115
                                 + 0.991944 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.794159 w_k1_ADM_s115
                                 + 0.702381 w_k1_AXP_s115
                                 + 0.739951 w_k1_DIS_s115
                                 + 0.752919 w_k1_FDX_s115
                                 + 1.20865 w_k1_IBM_s115
                                 + 0.880184 w_k1_INTC_s115
                                 + 0.976857 w_k1_KO_s115
                                 + 1.01273 w_k1_MAT_s115
                                 + 0.955458 w_k1_MSFT_s115
                                 + 0.878003 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.797693 w_k1_ADM_s115
                                 + 1.00425 w_k1_AXP_s115
                                 + 0.974825 w_k1_DIS_s115
                                 + 0.992064 w_k1_FDX_s115
                                 + 1.21524 w_k1_IBM_s115
                                 + 1.08043 w_k1_INTC_s115
                                 + 0.987197 w_k1_KO_s115
                                 + 1.03359 w_k1_MAT_s115
                                 + 1.18188 w_k1_MSFT_s115
                                 + 0.96903 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.27414 w_k1_ADM_s116
                                 + 1.15032 w_k1_AXP_s116
                                 + 1.10395 w_k1_DIS_s116
                                 + 1.10858 w_k1_FDX_s116
                                 + 1.09844 w_k1_IBM_s116
                                 + 0.969396 w_k1_INTC_s116
                                 + 1.00886 w_k1_KO_s116 + 1.40494 w_k1_MAT_s116
                                 + 1.12872 w_k1_MSFT_s116
                                 + 1.04106 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.839474 w_k1_ADM_s116
                                 + 0.77457 w_k1_AXP_s116
                                 + 0.657667 w_k1_DIS_s116
                                 + 0.744794 w_k1_FDX_s116
                                 + 0.888015 w_k1_IBM_s116
                                 + 0.746299 w_k1_INTC_s116
                                 + 0.678233 w_k1_KO_s116
                                 + 1.08446 w_k1_MAT_s116
                                 + 0.749683 w_k1_MSFT_s116
                                 + 0.902795 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26485 w_k1_ADM_s116
                                 + 1.74648 w_k1_AXP_s116
                                 + 1.69322 w_k1_DIS_s116 + 1.3938 w_k1_FDX_s116
                                 + 1.62478 w_k1_IBM_s116
                                 + 1.24306 w_k1_INTC_s116
                                 + 1.41643 w_k1_KO_s116 + 1.40176 w_k1_MAT_s116
                                 + 1.32017 w_k1_MSFT_s116
                                 + 1.33455 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.02837 w_k1_ADM_s116
                                 + 0.870329 w_k1_AXP_s116
                                 + 0.897078 w_k1_DIS_s116
                                 + 0.872961 w_k1_FDX_s116
                                 + 1.02727 w_k1_IBM_s116
                                 + 0.962823 w_k1_INTC_s116
                                 + 1.06497 w_k1_KO_s116 + 1.32662 w_k1_MAT_s116
                                 + 1.06413 w_k1_MSFT_s116
                                 + 1.05363 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.813599 w_k1_ADM_s116
                                 + 0.777672 w_k1_AXP_s116
                                 + 0.771451 w_k1_DIS_s116
                                 + 0.772518 w_k1_FDX_s116
                                 + 1.16206 w_k1_IBM_s116
                                 + 0.801034 w_k1_INTC_s116
                                 + 0.758603 w_k1_KO_s116
                                 + 1.07721 w_k1_MAT_s116
                                 + 0.766521 w_k1_MSFT_s116
                                 + 0.87543 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.08885 w_k1_ADM_s116
                                 + 0.814639 w_k1_AXP_s116
                                 + 0.789688 w_k1_DIS_s116
                                 + 0.917255 w_k1_FDX_s116
                                 + 0.665491 w_k1_IBM_s116
                                 + 0.854417 w_k1_INTC_s116
                                 + 0.791342 w_k1_KO_s116
                                 + 0.687362 w_k1_MAT_s116
                                 + 0.956089 w_k1_MSFT_s116
                                 + 0.967112 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.11745 w_k1_ADM_s116
                                 + 1.075 w_k1_AXP_s116 + 1.16699 w_k1_DIS_s116
                                 + 0.993621 w_k1_FDX_s116
                                 + 1.37452 w_k1_IBM_s116
                                 + 1.10198 w_k1_INTC_s116
                                 + 1.15898 w_k1_KO_s116 + 1.59692 w_k1_MAT_s116
                                 + 1.21472 w_k1_MSFT_s116
                                 + 1.22228 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.14627 w_k1_ADM_s116
                                 + 1.1079 w_k1_AXP_s116 + 1.18967 w_k1_DIS_s116
                                 + 1.22856 w_k1_FDX_s116
                                 + 0.827572 w_k1_IBM_s116
                                 + 1.17791 w_k1_INTC_s116
                                 + 0.910245 w_k1_KO_s116
                                 + 1.10899 w_k1_MAT_s116
                                 + 1.30003 w_k1_MSFT_s116
                                 + 1.04793 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.914076 w_k1_ADM_s117
                                 + 0.655677 w_k1_AXP_s117
                                 + 0.568933 w_k1_DIS_s117
                                 + 0.740704 w_k1_FDX_s117
                                 + 0.76771 w_k1_IBM_s117
                                 + 0.584838 w_k1_INTC_s117
                                 + 0.641437 w_k1_KO_s117
                                 + 0.709216 w_k1_MAT_s117
                                 + 0.707996 w_k1_MSFT_s117
                                 + 0.850509 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.25454 w_k1_ADM_s117
                                 + 1.35938 w_k1_AXP_s117
                                 + 1.70195 w_k1_DIS_s117
                                 + 1.41933 w_k1_FDX_s117
                                 + 0.837793 w_k1_IBM_s117
                                 + 1.13569 w_k1_INTC_s117
                                 + 1.25141 w_k1_KO_s117
                                 + 0.777963 w_k1_MAT_s117
                                 + 1.1786 w_k1_MSFT_s117 + 1.18626 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.05298 w_k1_ADM_s117
                                 + 0.965766 w_k1_AXP_s117
                                 + 1.00603 w_k1_DIS_s117
                                 + 1.04284 w_k1_FDX_s117
                                 + 0.84414 w_k1_IBM_s117
                                 + 1.06321 w_k1_INTC_s117
                                 + 1.05554 w_k1_KO_s117 + 0.81921 w_k1_MAT_s117
                                 + 0.909563 w_k1_MSFT_s117
                                 + 1.02442 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.15341 w_k1_ADM_s117
                                 + 0.984349 w_k1_AXP_s117
                                 + 1.03658 w_k1_DIS_s117
                                 + 0.97209 w_k1_FDX_s117
                                 + 1.06641 w_k1_IBM_s117
                                 + 0.994496 w_k1_INTC_s117
                                 + 0.937411 w_k1_KO_s117
                                 + 0.894009 w_k1_MAT_s117
                                 + 0.907716 w_k1_MSFT_s117
                                 + 1.02273 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.909347 w_k1_ADM_s117
                                 + 1.09774 w_k1_AXP_s117
                                 + 0.914361 w_k1_DIS_s117
                                 + 0.896818 w_k1_FDX_s117
                                 + 1.33892 w_k1_IBM_s117
                                 + 0.915119 w_k1_INTC_s117
                                 + 1.19769 w_k1_KO_s117 + 1.29127 w_k1_MAT_s117
                                 + 0.998257 w_k1_MSFT_s117
                                 + 1.06053 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.11149 w_k1_ADM_s117
                                 + 1.55988 w_k1_AXP_s117
                                 + 1.01109 w_k1_DIS_s117
                                 + 1.21315 w_k1_FDX_s117
                                 + 0.981283 w_k1_IBM_s117
                                 + 1.27146 w_k1_INTC_s117
                                 + 1.00163 w_k1_KO_s117 + 1.18842 w_k1_MAT_s117
                                 + 1.05884 w_k1_MSFT_s117
                                 + 1.14556 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.02816 w_k1_ADM_s117
                                 + 0.738959 w_k1_AXP_s117
                                 + 0.671619 w_k1_DIS_s117
                                 + 1.01196 w_k1_FDX_s117
                                 + 0.641868 w_k1_IBM_s117
                                 + 0.989649 w_k1_INTC_s117
                                 + 0.687469 w_k1_KO_s117
                                 + 0.534354 w_k1_MAT_s117
                                 + 0.952999 w_k1_MSFT_s117
                                 + 0.829015 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.39586 w_k1_ADM_s117
                                 + 1.66719 w_k1_AXP_s117
                                 + 1.76374 w_k1_DIS_s117
                                 + 1.38316 w_k1_FDX_s117
                                 + 1.56999 w_k1_IBM_s117
                                 + 1.40958 w_k1_INTC_s117
                                 + 1.75643 w_k1_KO_s117 + 1.64805 w_k1_MAT_s117
                                 + 1.58594 w_k1_MSFT_s117
                                 + 1.39269 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00839591356848445 w_k1_ADM_s000
                                 + 0.0116260612903247 w_k1_ADM_s001
                                 + 0.00879197223357173 w_k1_ADM_s002
                                 + 0.0253240113929826 w_k1_ADM_s003
                                 + 0.00764577477370678 w_k1_ADM_s004
                                 + 0.0282478363875966 w_k1_ADM_s005
                                 + 0.00509275048677373 w_k1_ADM_s006
                                 + 0.0177839455910839 w_k1_ADM_s007
                                 + 0.0371006535206635 w_k1_ADM_s010
                                 + 0.0352211185038916 w_k1_ADM_s011
                                 + 0.0817164907141884 w_k1_ADM_s012
                                 + 0.022404783309012 w_k1_ADM_s013
                                 + 0.0247100380362731 w_k1_ADM_s014
                                 + 0.0220157633733093 w_k1_ADM_s015
                                 + 0.0338002571886086 w_k1_ADM_s016
                                 + 0.017435847958538 w_k1_ADM_s017
                                 + 0.0303278860182443 w_k1_ADM_s100
                                 + 0.0537432961555751 w_k1_ADM_s101
                                 + 0.0187833351793688 w_k1_ADM_s102
                                 + 0.0995227412076808 w_k1_ADM_s103
                                 + 0.0235177853146837 w_k1_ADM_s104
                                 + 0.0352386101494253 w_k1_ADM_s105
                                 + 0.0291988480986275 w_k1_ADM_s106
                                 + 0.0391471331211264 w_k1_ADM_s107
                                 + 0.0126106828932062 w_k1_ADM_s110
                                 + 0.0387327466362525 w_k1_ADM_s111
                                 + 0.0606215767326309 w_k1_ADM_s112
                                 + 0.0157009542984107 w_k1_ADM_s113
                                 + 0.0374986606145974 w_k1_ADM_s114
                                 + 0.0537525432173544 w_k1_ADM_s115
                                 + 0.0534884561369564 w_k1_ADM_s116
                                 + 0.0251026362645227 w_k1_ADM_s117
                                 + 0.0082131786588345 w_k1_AXP_s000
                                 + 0.0112164479008206 w_k1_AXP_s001
                                 + 0.0103589762656962 w_k1_AXP_s002
                                 + 0.0277409659127212 w_k1_AXP_s003
                                 + 0.00596915239506659 w_k1_AXP_s004
                                 + 0.0274711453746743 w_k1_AXP_s005
                                 + 0.00569241813447346 w_k1_AXP_s006
                                 + 0.0172249543931942 w_k1_AXP_s007
                                 + 0.0365609153176887 w_k1_AXP_s010
                                 + 0.03946192618095 w_k1_AXP_s011
                                 + 0.0958081715819633 w_k1_AXP_s012
                                 + 0.025901206224416 w_k1_AXP_s013
                                 + 0.0263870498629698 w_k1_AXP_s014
                                 + 0.0241113397533516 w_k1_AXP_s015
                                 + 0.0336578916610547 w_k1_AXP_s016
                                 + 0.0175559150545464 w_k1_AXP_s017
                                 + 0.029737299636542 w_k1_AXP_s100
                                 + 0.0551849361185705 w_k1_AXP_s101
                                 + 0.0197617982086611 w_k1_AXP_s102
                                 + 0.0918579794119248 w_k1_AXP_s103
                                 + 0.024318084420313 w_k1_AXP_s104
                                 + 0.0322407447144686 w_k1_AXP_s105
                                 + 0.0298184889513831 w_k1_AXP_s106
                                 + 0.0343327578031849 w_k1_AXP_s107
                                 + 0.0144190403824783 w_k1_AXP_s110
                                 + 0.0420025184096288 w_k1_AXP_s111
                                 + 0.0658424525199266 w_k1_AXP_s112
                                 + 0.0146677471857581 w_k1_AXP_s113
                                 + 0.0409917807570393 w_k1_AXP_s114
                                 + 0.0577998728249584 w_k1_AXP_s115
                                 + 0.0496360077139582 w_k1_AXP_s116
                                 + 0.0244025363408993 w_k1_AXP_s117
                                 + 0.00924492258545376 w_k1_DIS_s000
                                 + 0.010754464568148 w_k1_DIS_s001
                                 + 0.0100861185082856 w_k1_DIS_s002
                                 + 0.0276532090219945 w_k1_DIS_s003
                                 + 0.006519917611467 w_k1_DIS_s004
                                 + 0.0296763112560351 w_k1_DIS_s005
                                 + 0.00602258272308564 w_k1_DIS_s006
                                 + 0.0180065785562055 w_k1_DIS_s007
                                 + 0.0389476786568264 w_k1_DIS_s010
                                 + 0.0395579422216995 w_k1_DIS_s011
                                 + 0.0843683523105121 w_k1_DIS_s012
                                 + 0.0248227817435884 w_k1_DIS_s013
                                 + 0.0247456499113598 w_k1_DIS_s014
                                 + 0.0228553600262234 w_k1_DIS_s015
                                 + 0.0357671937459287 w_k1_DIS_s016
                                 + 0.018263466899204 w_k1_DIS_s017
                                 + 0.0298962336942683 w_k1_DIS_s100
                                 + 0.0529866710018906 w_k1_DIS_s101
                                 + 0.0210281850256897 w_k1_DIS_s102
                                 + 0.0886053319600137 w_k1_DIS_s103
                                 + 0.0239455902492639 w_k1_DIS_s104
                                 + 0.036572289949886 w_k1_DIS_s105
                                 + 0.0287709921884483 w_k1_DIS_s106
                                 + 0.0335813203520533 w_k1_DIS_s107
                                 + 0.0146460895293423 w_k1_DIS_s110
                                 + 0.0435793446578625 w_k1_DIS_s111
                                 + 0.066986163703414 w_k1_DIS_s112
                                 + 0.0164465719399675 w_k1_DIS_s113
                                 + 0.0392031600953873 w_k1_DIS_s114
                                 + 0.0600723757712536 w_k1_DIS_s115
                                 + 0.049815354018174 w_k1_DIS_s116
                                 + 0.0236438093350609 w_k1_DIS_s117
                                 + 0.00845715221214577 w_k1_FDX_s000
                                 + 0.0108594344783185 w_k1_FDX_s001
                                 + 0.00918278694648415 w_k1_FDX_s002
                                 + 0.0265942855993761 w_k1_FDX_s003
                                 + 0.00634201112811031 w_k1_FDX_s004
                                 + 0.02824082119916 w_k1_FDX_s005
                                 + 0.00583302557778794 w_k1_FDX_s006
                                 + 0.0169084238320936 w_k1_FDX_s007
                                 + 0.0404432733489789 w_k1_FDX_s010
                                 + 0.0398247632661851 w_k1_FDX_s011
                                 + 0.0829489239584296 w_k1_FDX_s012
                                 + 0.0248452477642573 w_k1_FDX_s013
                                 + 0.0253998048139837 w_k1_FDX_s014
                                 + 0.0228391073939433 w_k1_FDX_s015
                                 + 0.0341841046828209 w_k1_FDX_s016
                                 + 0.0184561390342676 w_k1_FDX_s017
                                 + 0.0298301727958466 w_k1_FDX_s100
                                 + 0.0504072042712543 w_k1_FDX_s101
                                 + 0.0199991154837967 w_k1_FDX_s102
                                 + 0.0883203240711609 w_k1_FDX_s103
                                 + 0.0245819192075404 w_k1_FDX_s104
                                 + 0.0343178783373606 w_k1_FDX_s105
                                 + 0.0282937190827452 w_k1_FDX_s106
                                 + 0.0349206983421129 w_k1_FDX_s107
                                 + 0.014648256646575 w_k1_FDX_s110
                                 + 0.0433024727368304 w_k1_FDX_s111
                                 + 0.0670848832566722 w_k1_FDX_s112
                                 + 0.0167080847060891 w_k1_FDX_s113
                                 + 0.0401117669404271 w_k1_FDX_s114
                                 + 0.0602677753560198 w_k1_FDX_s115
                                 + 0.049379263579945 w_k1_FDX_s116
                                 + 0.0237774718630662 w_k1_FDX_s117
                                 + 0.00866691380596891 w_k1_IBM_s000
                                 + 0.0106019947844931 w_k1_IBM_s001
                                 + 0.00981711292395389 w_k1_IBM_s002
                                 + 0.0266808925856973 w_k1_IBM_s003
                                 + 0.00674945089054584 w_k1_IBM_s004
                                 + 0.0285991037666072 w_k1_IBM_s005
                                 + 0.00651114210198856 w_k1_IBM_s006
                                 + 0.0171829119836904 w_k1_IBM_s007
                                 + 0.0366228478383294 w_k1_IBM_s010
                                 + 0.0394038962900379 w_k1_IBM_s011
                                 + 0.0835914913861948 w_k1_IBM_s012
                                 + 0.0261598089621986 w_k1_IBM_s013
                                 + 0.0254276989134948 w_k1_IBM_s014
                                 + 0.0232462435799543 w_k1_IBM_s015
                                 + 0.0349577319518787 w_k1_IBM_s016
                                 + 0.0185550981310335 w_k1_IBM_s017
                                 + 0.0294083020464031 w_k1_IBM_s100
                                 + 0.0497123971105812 w_k1_IBM_s101
                                 + 0.0206214272730867 w_k1_IBM_s102
                                 + 0.0880762587777072 w_k1_IBM_s103
                                 + 0.024157686299371 w_k1_IBM_s104
                                 + 0.0350999349751267 w_k1_IBM_s105
                                 + 0.0287696674392051 w_k1_IBM_s106
                                 + 0.0335008547537808 w_k1_IBM_s107
                                 + 0.0148979496791518 w_k1_IBM_s110
                                 + 0.0437144107674813 w_k1_IBM_s111
                                 + 0.0668878591239334 w_k1_IBM_s112
                                 + 0.0164836955425223 w_k1_IBM_s113
                                 + 0.0400992383247832 w_k1_IBM_s114
                                 + 0.0622644178783462 w_k1_IBM_s115
                                 + 0.0497373673176076 w_k1_IBM_s116
                                 + 0.0231562475318542 w_k1_IBM_s117
                                 + 0.00711542868108137 w_k1_INTC_s000
                                 + 0.00944470631196937 w_k1_INTC_s001
                                 + 0.00832840736599138 w_k1_INTC_s002
                                 + 0.0272152917087693 w_k1_INTC_s003
                                 + 0.0058991403961551 w_k1_INTC_s004
                                 + 0.0269636967176201 w_k1_INTC_s005
                                 + 0.00638530671341421 w_k1_INTC_s006
                                 + 0.0189976013275237 w_k1_INTC_s007
                                 + 0.0385498654014397 w_k1_INTC_s010
                                 + 0.038799481263155 w_k1_INTC_s011
                                 + 0.0792370985812714 w_k1_INTC_s012
                                 + 0.0261027323286752 w_k1_INTC_s013
                                 + 0.0227097447870271 w_k1_INTC_s014
                                 + 0.0236589724213828 w_k1_INTC_s015
                                 + 0.0322001433319934 w_k1_INTC_s016
                                 + 0.0165393169421055 w_k1_INTC_s017
                                 + 0.0303465182834627 w_k1_INTC_s100
                                 + 0.0466702980261502 w_k1_INTC_s101
                                 + 0.0222397338574658 w_k1_INTC_s102
                                 + 0.0799430372037327 w_k1_INTC_s103
                                 + 0.0287813932580679 w_k1_INTC_s104
                                 + 0.0322618167176359 w_k1_INTC_s105
                                 + 0.0323666309145816 w_k1_INTC_s106
                                 + 0.0330307210341979 w_k1_INTC_s107
                                 + 0.0138555688433703 w_k1_INTC_s110
                                 + 0.043011549158861 w_k1_INTC_s111
                                 + 0.0597042980641092 w_k1_INTC_s112
                                 + 0.0184904688945044 w_k1_INTC_s113
                                 + 0.0366996350738306 w_k1_INTC_s114
                                 + 0.063777517674413 w_k1_INTC_s115
                                 + 0.0479405163187424 w_k1_INTC_s116
                                 + 0.0229746242295665 w_k1_INTC_s117
                                 + 0.00855969762563165 w_k1_KO_s000
                                 + 0.00978719198692758 w_k1_KO_s001
                                 + 0.0107500640169267 w_k1_KO_s002
                                 + 0.0267846154492271 w_k1_KO_s003
                                 + 0.006440035563842 w_k1_KO_s004
                                 + 0.0313135407832047 w_k1_KO_s005
                                 + 0.00614757156543784 w_k1_KO_s006
                                 + 0.0174383500418018 w_k1_KO_s007
                                 + 0.0383818168223889 w_k1_KO_s010
                                 + 0.0347694283252317 w_k1_KO_s011
                                 + 0.0873003585162736 w_k1_KO_s012
                                 + 0.025623187883269 w_k1_KO_s013
                                 + 0.0215876302471694 w_k1_KO_s014
                                 + 0.0239805568359967 w_k1_KO_s015
                                 + 0.0343313516446814 w_k1_KO_s016
                                 + 0.0161711119038324 w_k1_KO_s017
                                 + 0.0296018901001159 w_k1_KO_s100
                                 + 0.0509997442505738 w_k1_KO_s101
                                 + 0.0205042145454543 w_k1_KO_s102
                                 + 0.0784317646249699 w_k1_KO_s103
                                 + 0.0243758889294244 w_k1_KO_s104
                                 + 0.0349473989730393 w_k1_KO_s105
                                 + 0.0289073892546702 w_k1_KO_s106
                                 + 0.0362154518382183 w_k1_KO_s107
                                 + 0.0153667062147964 w_k1_KO_s110
                                 + 0.0400537332406295 w_k1_KO_s111
                                 + 0.0673936568019775 w_k1_KO_s112
                                 + 0.0152841006867293 w_k1_KO_s113
                                 + 0.0378230971516123 w_k1_KO_s114
                                 + 0.0571597530440085 w_k1_KO_s115
                                 + 0.0465047269473582 w_k1_KO_s116
                                 + 0.0233394230655034 w_k1_KO_s117
                                 + 0.00967320824992005 w_k1_MAT_s000
                                 + 0.0104280941185811 w_k1_MAT_s001
                                 + 0.0146262779016957 w_k1_MAT_s002
                                 + 0.0252801640633648 w_k1_MAT_s003
                                 + 0.00574175980951295 w_k1_MAT_s004
                                 + 0.0284624370835459 w_k1_MAT_s005
                                 + 0.00599051387434216 w_k1_MAT_s006
                                 + 0.0198954095457777 w_k1_MAT_s007
                                 + 0.0336368408206777 w_k1_MAT_s010
                                 + 0.041185529557962 w_k1_MAT_s011
                                 + 0.0819337039027741 w_k1_MAT_s012
                                 + 0.0216164577997402 w_k1_MAT_s013
                                 + 0.0249510628156341 w_k1_MAT_s014
                                 + 0.0268380749925136 w_k1_MAT_s015
                                 + 0.0437013026850161 w_k1_MAT_s016
                                 + 0.0204776084891236 w_k1_MAT_s017
                                 + 0.0290659623852398 w_k1_MAT_s100
                                 + 0.066881481285423 w_k1_MAT_s101
                                 + 0.01972017231727 w_k1_MAT_s102
                                 + 0.10583258471792 w_k1_MAT_s103
                                 + 0.0254720303130895 w_k1_MAT_s104
                                 + 0.0365023091445191 w_k1_MAT_s105
                                 + 0.027206222968168 w_k1_MAT_s106
                                 + 0.0334296439786644 w_k1_MAT_s107
                                 + 0.0195676147074992 w_k1_MAT_s110
                                 + 0.0402065130851261 w_k1_MAT_s111
                                 + 0.0593574965176536 w_k1_MAT_s112
                                 + 0.0160816983054207 w_k1_MAT_s113
                                 + 0.0392327442856226 w_k1_MAT_s114
                                 + 0.0610175791159441 w_k1_MAT_s115
                                 + 0.0575687826791434 w_k1_MAT_s116
                                 + 0.0218870490989411 w_k1_MAT_s117
                                 + 0.00860213892243991 w_k1_MSFT_s000
                                 + 0.00955132512352283 w_k1_MSFT_s001
                                 + 0.00880573813350087 w_k1_MSFT_s002
                                 + 0.024096015450403 w_k1_MSFT_s003
                                 + 0.00621190812657537 w_k1_MSFT_s004
                                 + 0.0287316672829662 w_k1_MSFT_s005
                                 + 0.00585360006362819 w_k1_MSFT_s006
                                 + 0.0182109218378189 w_k1_MSFT_s007
                                 + 0.0337148693252978 w_k1_MSFT_s010
                                 + 0.0373874371418655 w_k1_MSFT_s011
                                 + 0.0782355735414514 w_k1_MSFT_s012
                                 + 0.0259538145978233 w_k1_MSFT_s013
                                 + 0.0237453630044027 w_k1_MSFT_s014
                                 + 0.0240159105593911 w_k1_MSFT_s015
                                 + 0.0314549480703744 w_k1_MSFT_s016
                                 + 0.016340282500397 w_k1_MSFT_s017
                                 + 0.0305640457290398 w_k1_MSFT_s100
                                 + 0.0483745644667312 w_k1_MSFT_s101
                                 + 0.0191392595453969 w_k1_MSFT_s102
                                 + 0.0852387656445327 w_k1_MSFT_s103
                                 + 0.0242429647888286 w_k1_MSFT_s104
                                 + 0.0317627711750125 w_k1_MSFT_s105
                                 + 0.0283777533404394 w_k1_MSFT_s106
                                 + 0.0350618470001071 w_k1_MSFT_s107
                                 + 0.0146151416695424 w_k1_MSFT_s110
                                 + 0.0398983531750573 w_k1_MSFT_s111
                                 + 0.0662916567255759 w_k1_MSFT_s112
                                 + 0.0165027177313144 w_k1_MSFT_s113
                                 + 0.0375666111326535 w_k1_MSFT_s114
                                 + 0.0661482188871875 w_k1_MSFT_s115
                                 + 0.0527109211952363 w_k1_MSFT_s116
                                 + 0.0227372831559522 w_k1_MSFT_s117
                                 + 0.00896454087601308 w_k1_PG_s000
                                 + 0.0106902575426387 w_k1_PG_s001
                                 + 0.00904653733596646 w_k1_PG_s002
                                 + 0.0260714427532152 w_k1_PG_s003
                                 + 0.00647311503088514 w_k1_PG_s004
                                 + 0.0268287912301539 w_k1_PG_s005
                                 + 0.00546006362966154 w_k1_PG_s006
                                 + 0.0177435908382625 w_k1_PG_s007
                                 + 0.0389909214325524 w_k1_PG_s010
                                 + 0.0352643266953435 w_k1_PG_s011
                                 + 0.0811951530290562 w_k1_PG_s012
                                 + 0.0249214056436588 w_k1_PG_s013
                                 + 0.0251989071788336 w_k1_PG_s014
                                 + 0.0231841842968047 w_k1_PG_s015
                                 + 0.0335181744911985 w_k1_PG_s016
                                 + 0.0201731439622411 w_k1_PG_s017
                                 + 0.028879783328171 w_k1_PG_s100
                                 + 0.0513036686988668 w_k1_PG_s101
                                 + 0.0202019257892931 w_k1_PG_s102
                                 + 0.0878024171525482 w_k1_PG_s103
                                 + 0.0259321867320675 w_k1_PG_s104
                                 + 0.032100770301034 w_k1_PG_s105
                                 + 0.0276144990214696 w_k1_PG_s106
                                 + 0.0336396263913283 w_k1_PG_s107
                                 + 0.0145670441098782 w_k1_PG_s110
                                 + 0.0415350158172648 w_k1_PG_s111
                                 + 0.0710816564494829 w_k1_PG_s112
                                 + 0.0150613523044758 w_k1_PG_s113
                                 + 0.0412979497364359 w_k1_PG_s114
                                 + 0.0578060485241072 w_k1_PG_s115
                                 + 0.0510230502093272 w_k1_PG_s116
                                 + 0.0239129747933173 w_k1_PG_s117 >= 1.16
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
