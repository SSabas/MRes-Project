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
 w_k1_ADM_s0_return_constraint:  1.14007 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.901687 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.913941 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.22576 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.895701 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.12257 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.17155 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.11363 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.06309 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.90095 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.29661 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.881509 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.887874 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.21534 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.913703 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.08908 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.91024 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.841554 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.70301 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.905269 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.19483 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.23622 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.862865 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.757946 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.32744 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.05797 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.850388 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.1415 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.02721 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.611074 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.12682 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.34447 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.942167 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.16487 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.779489 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.34479 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.94649 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.918745 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.17617 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.934201 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.876324 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24855 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.809072 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.1165 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.20008 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.15396 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.62092 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.927485 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.47168 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.789503 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.742115 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.12185 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.881818 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.16031 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.07596 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.786079 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.626113 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.869707 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.5932 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.57748 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.796964 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.741837 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.31138 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.874492 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.998504 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.16246 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.24163 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.601881 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.10562 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.56521 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.974479 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.35527 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.901464 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.90232 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.984179 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.709669 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26977 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.911938 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.838179 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.35796 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.72101 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.19228 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.16521 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.09851 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.49954 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.913072 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.6222 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.777322 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.720331 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.10822 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.809351 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.206 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.94714 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.737426 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.570329 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.992407 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.62008 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.51057 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.832022 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.732699 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.63527 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.05925 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.92952 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.11047 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.36263 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.577777 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.33668 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.19699 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.993472 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.32901 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.809754 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 2.05696 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.873221 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.654423 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20447 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.905984 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.875591 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.21342 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.820181 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.1345 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.04798 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.14297 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.23206 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.87328 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.35954 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.857215 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.877957 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.1624 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.932446 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07233 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.864359 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.749704 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.690564 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.990065 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.23981 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.47765 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.791353 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.872685 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.46784 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.15896 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.901933 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.13079 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.20333 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.706914 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.11947 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.31376 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.975879 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.06356 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.827493 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.45503 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.06878 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.80639 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.04797 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.00488 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.986494 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.14817 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.884498 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.15211 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.32265 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.914369 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.89906 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.19616 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.02023 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.869354 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.568637 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.622558 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.581107 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.1242 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.53089 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.934157 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.613298 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.891678 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.80017 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.02887 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.02542 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.656875 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.867772 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.637128 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.36291 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.91603 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.59156 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.733948 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.10009 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.867876 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.794756 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.79573 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.18461 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.46654 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.645985 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.647344 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.08979 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.952887 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.910182 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.26417 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.818016 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.10753 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.08488 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.971774 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.18531 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.05038 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.28734 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.963109 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.974381 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.00859 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.799228 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.02891 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.963665 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.826133 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.623881 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.00349 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.08989 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.23662 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.924468 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07828 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.30183 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.07759 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.05294 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.994811 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.14624 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.802391 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.05978 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.21382 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.03202 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.943221 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.934633 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.4483 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.907907 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.776294 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.08371 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.953703 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.935479 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.102 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.798695 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.13608 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.21601 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.0406 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.48484 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.01342 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.2475 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.928609 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.766085 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.796787 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.796322 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.21116 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.31277 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.00918 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.771224 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.07866 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.72179 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.40479 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.944746 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.737528 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.29617 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.863814 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.15022 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.0276 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.35743 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.696338 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.22747 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.956379 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.970843 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.48682 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.10124 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.57618 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.752111 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.674538 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.14103 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.04552 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.00207 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.18833 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.893375 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.973864 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.2974 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.886245 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.99931 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.00142 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.07579 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.772884 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.38428 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.677116 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.493074 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.11333 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.37216 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.838519 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.594187 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.847972 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.14134 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.2552 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.71415 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.545986 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.743038 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.649734 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.18594 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.990243 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.63565 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.497525 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.32085 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.28955 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.04133 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 2.1742 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.37361 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.96486 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.838334 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.798499 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15982 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.914391 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.954154 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.22062 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.857399 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.16963 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.932345 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.01997 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.07789 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.825152 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.16806 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.819743 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.798592 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.04021 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.851057 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.09252 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.998485 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.799701 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.712244 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.994514 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.26634 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.22234 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.90289 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.88582 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.37149 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.989824 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.997482 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.982417 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.19188 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.685711 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.13182 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.18615 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.953205 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.997478 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.854616 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.36288 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.943303 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.830821 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.07523 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.937045 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.925648 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.12021 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.877446 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.13549 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.14721 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.10252 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.24409 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.985753 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.1277 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.94534 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.888613 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.07988 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.867474 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.14541 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.932719 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.933508 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.745648 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.985156 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.36085 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.36009 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.893454 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.881651 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.17291 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.00507 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.975031 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.11864 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.18338 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.755516 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.05578 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.09324 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.978347 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.13666 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.913314 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.22536 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.918422 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.769303 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.243000256944345 z_k1_s0000
                                 - 0.560000934727945 z_k1_s0001
                                 - 0.199000104215667 z_k1_s0002
                                 - 0.526001611369966 z_k1_s0003
                                 - 0.227000150952064 z_k1_s0004
                                 - 0.449999859155772 z_k1_s0005
                                 - 0.158000006048065 z_k1_s0006
                                 - 0.411998983048674 z_k1_s0007
                                 - 0.202000409543886 z_k1_s0010
                                 - 0.583001522939576 z_k1_s0011
                                 - 0.29500117549322 z_k1_s0012
                                 - 0.289999790601351 z_k1_s0013
                                 - 0.199000411710079 z_k1_s0014
                                 - 0.184000422541044 z_k1_s0015
                                 - 0.234000386437829 z_k1_s0016
                                 - 0.790002206573579 z_k1_s0017
                                 - 0.613999645762674 z_k1_s0020
                                 - 0.23399973628172 z_k1_s0021
                                 - 0.130999696317762 z_k1_s0022
                                 - 0.139999782366924 z_k1_s0023
                                 - 0.544999652984829 z_k1_s0024
                                 - 0.130999696317762 z_k1_s0025
                                 - 0.397000683372965 z_k1_s0026
                                 - 0.309000620091177 z_k1_s0027
                                 - 0.317999384926093 z_k1_s0030
                                 - 0.573997736797283 z_k1_s0031
                                 - 0.343999339304946 z_k1_s0032
                                 - 0.812996775057865 z_k1_s0033
                                 - 0.346999305494873 z_k1_s0034
                                 - 0.291999430547239 z_k1_s0035
                                 - 0.760998350697086 z_k1_s0036
                                 - 0.26199939754874 z_k1_s0037
                                 - 0.31000002963055 z_k1_s0040
                                 - 0.270999893025469 z_k1_s0041
                                 - 0.252998996131145 z_k1_s0042
                                 - 0.345000207814575 z_k1_s0043
                                 - 0.18600001777833 z_k1_s0044
                                 - 0.214999824730942 z_k1_s0045
                                 - 0.169999967294192 z_k1_s0046
                                 - 0.417998906999059 z_k1_s0047
                                 - 0.147999725958771 z_k1_s0050
                                 - 0.384997731210874 z_k1_s0051
                                 - 0.192999517739119 z_k1_s0052
                                 - 0.168999545243145 z_k1_s0053
                                 - 0.549999891864752 z_k1_s0054
                                 - 0.65499898828662 z_k1_s0055
                                 - 0.559997269578012 z_k1_s0056
                                 - 0.473000345622578 z_k1_s0057
                                 - 0.776000023399699 z_k1_s0060
                                 - 1.0139985489381 z_k1_s0061
                                 - 0.40800020985471 z_k1_s0062
                                 - 0.369000644703683 z_k1_s0063
                                 - 1.1530034848685 z_k1_s0064
                                 - 0.292000155557789 z_k1_s0065
                                 - 0.675999032625398 z_k1_s0066
                                 - 0.786999311133921 z_k1_s0067
                                 - 0.155000420559716 z_k1_s0070
                                 - 0.0819999417383268 z_k1_s0071
                                 - 0.183000337299697 z_k1_s0072
                                 - 0.208999796058718 z_k1_s0073
                                 - 0.152999962578757 z_k1_s0074
                                 - 0.279000237408014 z_k1_s0075
                                 - 0.131000120782965 z_k1_s0076
                                 - 0.106999950906082 z_k1_s0077
                                 - 0.656998565099479 z_k1_s0100
                                 - 0.292999652943781 z_k1_s0101
                                 - 0.222999841298463 z_k1_s0102
                                 - 0.197999831139455 z_k1_s0103
                                 - 0.157999977484971 z_k1_s0104
                                 - 0.175999789679541 z_k1_s0105
                                 - 0.723000586478396 z_k1_s0106
                                 - 0.281999767713765 z_k1_s0107
                                 - 0.213000555702269 z_k1_s0110
                                 - 0.354001869088378 z_k1_s0111
                                 - 0.522002039365398 z_k1_s0112
                                 - 0.309002012264759 z_k1_s0113
                                 - 0.301001626583736 z_k1_s0114
                                 - 0.172000627418346 z_k1_s0115
                                 - 0.448000454072052 z_k1_s0116
                                 - 0.324001083536581 z_k1_s0117
                                 - 0.275000384539906 z_k1_s0120
                                 - 0.162000528376311 z_k1_s0121
                                 - 0.0780000430403932 z_k1_s0122
                                 - 0.179000311613263 z_k1_s0123
                                 - 0.111000011365163 z_k1_s0124
                                 - 0.091000006980448 z_k1_s0125
                                 - 0.208999929090238 z_k1_s0126
                                 - 0.192000145853286 z_k1_s0127
                                 - 0.142000533644469 z_k1_s0130
                                 - 0.262000977153855 z_k1_s0131
                                 - 0.421001516266135 z_k1_s0132
                                 - 0.119000391275818 z_k1_s0133
                                 - 0.397000968662773 z_k1_s0134
                                 - 0.116000411075683 z_k1_s0135
                                 - 0.161000467078846 z_k1_s0136
                                 - 0.14700044181117 z_k1_s0137
                                 - 0.036000033634444 z_k1_s0140
                                 - 0.103000001292863 z_k1_s0141
                                 - 0.0510000352654455 z_k1_s0142
                                 - 0.118000448724472 z_k1_s0143
                                 - 0.115000210637948 z_k1_s0144
                                 - 0.124000181896508 z_k1_s0145
                                 - 0.0370000386965175 z_k1_s0146
                                 - 0.158999987568575 z_k1_s0147
                                 - 0.293999611184237 z_k1_s0150
                                 - 0.46899836031933 z_k1_s0151
                                 - 0.268999455379755 z_k1_s0152
                                 - 0.44599817683927 z_k1_s0153
                                 - 0.148999777916541 z_k1_s0154
                                 - 0.496999310859118 z_k1_s0155
                                 - 0.528000721634742 z_k1_s0156
                                 - 0.692999274648255 z_k1_s0157
                                 - 0.207000931809342 z_k1_s0160
                                 - 0.176001000829027 z_k1_s0161
                                 - 0.135000565988543 z_k1_s0162
                                 - 0.252000161068539 z_k1_s0163
                                 - 0.118000412988934 z_k1_s0164
                                 - 0.0930004325433807 z_k1_s0165
                                 - 0.429002139707112 z_k1_s0166
                                 - 0.189000920304447 z_k1_s0167
                                 - 0.160999562089344 z_k1_s0170
                                 - 0.39700017228309 z_k1_s0171
                                 - 0.0990000481370958 z_k1_s0172
                                 - 0.0850000750510599 z_k1_s0173
                                 - 0.125000034839753 z_k1_s0174
                                 - 0.145999866068739 z_k1_s0175
                                 - 0.0990000481370958 z_k1_s0176
                                 - 0.171999595971259 z_k1_s0177
                                 - 0.237000902573918 z_k1_s1000
                                 - 0.416000638002786 z_k1_s1001
                                 - 0.150000076656712 z_k1_s1002
                                 - 0.640001159388972 z_k1_s1003
                                 - 0.182999959425136 z_k1_s1004
                                 - 0.189000064219445 z_k1_s1005
                                 - 0.363000328853301 z_k1_s1006
                                 - 0.134000105472011 z_k1_s1007
                                 - 0.621000763988006 z_k1_s1010
                                 - 0.265000678662696 z_k1_s1011
                                 - 0.52599963955592 z_k1_s1012
                                 - 1.09300033711019 z_k1_s1013
                                 - 0.883003101430498 z_k1_s1014
                                 - 0.469002592105161 z_k1_s1015
                                 - 0.328000756168726 z_k1_s1016
                                 - 0.349000933137756 z_k1_s1017
                                 - 0.0879999623867715 z_k1_s1020
                                 - 0.144000061505654 z_k1_s1021
                                 - 0.251999769234818 z_k1_s1022
                                 - 0.33599969231309 z_k1_s1023
                                 - 0.125000025190318 z_k1_s1024
                                 - 0.128000054671693 z_k1_s1025
                                 - 0.2200006579671 z_k1_s1026
                                 - 0.211000719923011 z_k1_s1027
                                 - 0.252000375285694 z_k1_s1030
                                 - 0.321000204399352 z_k1_s1031
                                 - 0.464999597918901 z_k1_s1032
                                 - 0.834001898760475 z_k1_s1033
                                 - 0.410001705156426 z_k1_s1034
                                 - 0.717999099035778 z_k1_s1035
                                 - 0.311000134778435 z_k1_s1036
                                 - 0.520001321685331 z_k1_s1037
                                 - 0.240999638352683 z_k1_s1040
                                 - 0.253999561446497 z_k1_s1041
                                 - 0.452000163564545 z_k1_s1042
                                 - 0.495998248038833 z_k1_s1043
                                 - 0.349998875293335 z_k1_s1044
                                 - 0.798998820144339 z_k1_s1045
                                 - 0.247999431418875 z_k1_s1046
                                 - 0.233999537887033 z_k1_s1047
                                 - 0.276999357233179 z_k1_s1050
                                 - 0.244999344892407 z_k1_s1051
                                 - 0.590997545644976 z_k1_s1052
                                 - 0.428999054602868 z_k1_s1053
                                 - 0.714998424338113 z_k1_s1054
                                 - 0.247999156393642 z_k1_s1055
                                 - 0.233999458056184 z_k1_s1056
                                 - 0.15099947120343 z_k1_s1057
                                 - 0.663999796660655 z_k1_s1060
                                 - 0.587000365692214 z_k1_s1061
                                 - 0.326999747135065 z_k1_s1062
                                 - 0.536998853739586 z_k1_s1063
                                 - 0.949997683114864 z_k1_s1064
                                 - 0.973001276051679 z_k1_s1065
                                 - 0.787001239505215 z_k1_s1066
                                 - 0.348999584554619 z_k1_s1067
                                 - 0.343999389778573 z_k1_s1070
                                 - 0.165000520794269 z_k1_s1071
                                 - 0.118999951603497 z_k1_s1072
                                 - 0.313999295088706 z_k1_s1073
                                 - 0.0689998997869625 z_k1_s1074
                                 - 0.132999889792191 z_k1_s1075
                                 - 0.209999072830411 z_k1_s1076
                                 - 0.236000320893978 z_k1_s1077
                                 - 0.754000981318747 z_k1_s1100
                                 - 0.536000174321837 z_k1_s1101
                                 - 0.670001026402671 z_k1_s1102
                                 - 0.800001418630145 z_k1_s1103
                                 - 1.33399974602455 z_k1_s1104
                                 - 0.719002617104464 z_k1_s1105
                                 - 0.668998485938208 z_k1_s1106
                                 - 0.986001780801668 z_k1_s1107
                                 - 0.372000863291524 z_k1_s1110
                                 - 0.220000134484059 z_k1_s1111
                                 - 0.579000059719274 z_k1_s1112
                                 - 0.508000196644957 z_k1_s1113
                                 - 0.205000196496543 z_k1_s1114
                                 - 0.37100123282586 z_k1_s1115
                                 - 0.125000005229546 z_k1_s1116
                                 - 0.230000093142404 z_k1_s1117
                                 - 0.274000349296276 z_k1_s1120
                                 - 0.269000464980805 z_k1_s1121
                                 - 0.66200252019093 z_k1_s1122
                                 - 0.322000353526209 z_k1_s1123
                                 - 0.276000303022464 z_k1_s1124
                                 - 0.564999933664685 z_k1_s1125
                                 - 0.232000206244909 z_k1_s1126
                                 - 1.11600167764793 z_k1_s1127
                                 - 0.391001132886698 z_k1_s1130
                                 - 0.600003943402815 z_k1_s1131
                                 - 0.970003560626448 z_k1_s1132
                                 - 0.255000564942998 z_k1_s1133
                                 - 0.266000829812604 z_k1_s1134
                                 - 0.816000741254522 z_k1_s1135
                                 - 0.8620007176878 z_k1_s1136
                                 - 0.284000859234432 z_k1_s1137
                                 - 0.283999016145169 z_k1_s1140
                                 - 0.290999453647593 z_k1_s1141
                                 - 1.22599948561735 z_k1_s1142
                                 - 0.295999215150702 z_k1_s1143
                                 - 1.29399822565467 z_k1_s1144
                                 - 0.45699880873262 z_k1_s1145
                                 - 0.634998362823179 z_k1_s1146
                                 - 1.02699531302778 z_k1_s1147
                                 - 0.416999318891311 z_k1_s1150
                                 - 0.132000084078051 z_k1_s1151
                                 - 0.249999908783999 z_k1_s1152
                                 - 0.195000149251686 z_k1_s1153
                                 - 0.220999654884855 z_k1_s1154
                                 - 0.49199929152651 z_k1_s1155
                                 - 0.168000026862914 z_k1_s1156
                                 - 0.329000144439942 z_k1_s1157
                                 - 0.713003148221051 z_k1_s1160
                                 - 0.570001523085738 z_k1_s1161
                                 - 0.610000572072916 z_k1_s1162
                                 - 0.58400119023125 z_k1_s1163
                                 - 0.231000035407718 z_k1_s1164
                                 - 0.76700186435374 z_k1_s1165
                                 - 0.537002307671317 z_k1_s1166
                                 - 1.01300104063103 z_k1_s1167
                                 - 0.456001396561222 z_k1_s1170
                                 - 0.522001389479965 z_k1_s1171
                                 - 0.875002092194614 z_k1_s1172
                                 - 0.465001582973445 z_k1_s1173
                                 - 0.96100531300474 z_k1_s1174
                                 - 0.84600456690914 z_k1_s1175
                                 - 0.317001134869381 z_k1_s1176
                                 - 1.44700242342493 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.737163 w_k1_ADM_s000
                                 + 0.944868 w_k1_AXP_s000
                                 + 0.648908 w_k1_DIS_s000
                                 + 0.748783 w_k1_FDX_s000
                                 + 0.863301 w_k1_IBM_s000
                                 + 0.710464 w_k1_INTC_s000
                                 + 0.80042 w_k1_KO_s000 + 1.01459 w_k1_MAT_s000
                                 + 0.716414 w_k1_MSFT_s000
                                 + 0.79764 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18313 w_k1_ADM_s000
                                 + 1.38728 w_k1_AXP_s000
                                 + 1.12299 w_k1_DIS_s000
                                 + 1.11087 w_k1_FDX_s000
                                 + 1.06602 w_k1_IBM_s000
                                 + 1.02223 w_k1_INTC_s000
                                 + 1.00896 w_k1_KO_s000 + 1.16786 w_k1_MAT_s000
                                 + 1.03843 w_k1_MSFT_s000
                                 + 1.07217 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.33267 w_k1_ADM_s000
                                 + 1.63137 w_k1_AXP_s000
                                 + 1.46294 w_k1_DIS_s000
                                 + 1.43175 w_k1_FDX_s000 + 1.4318 w_k1_IBM_s000
                                 + 1.13407 w_k1_INTC_s000
                                 + 1.38914 w_k1_KO_s000 + 1.77496 w_k1_MAT_s000
                                 + 1.23811 w_k1_MSFT_s000
                                 + 1.34953 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.673208 w_k1_ADM_s000
                                 + 1.03186 w_k1_AXP_s000
                                 + 0.865324 w_k1_DIS_s000
                                 + 0.847554 w_k1_FDX_s000
                                 + 1.14481 w_k1_IBM_s000
                                 + 0.960387 w_k1_INTC_s000
                                 + 0.949954 w_k1_KO_s000 + 1.2174 w_k1_MAT_s000
                                 + 0.947366 w_k1_MSFT_s000
                                 + 0.864713 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.76948 w_k1_ADM_s000
                                 + 1.02574 w_k1_AXP_s000
                                 + 1.09472 w_k1_DIS_s000
                                 + 0.86801 w_k1_FDX_s000
                                 + 1.67681 w_k1_IBM_s000
                                 + 0.959962 w_k1_INTC_s000
                                 + 1.26917 w_k1_KO_s000 + 1.75619 w_k1_MAT_s000
                                 + 0.930672 w_k1_MSFT_s000
                                 + 1.12293 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.20166 w_k1_ADM_s000
                                 + 1.19514 w_k1_AXP_s000
                                 + 1.12549 w_k1_DIS_s000
                                 + 1.14349 w_k1_FDX_s000
                                 + 0.840283 w_k1_IBM_s000
                                 + 0.977419 w_k1_INTC_s000
                                 + 0.940994 w_k1_KO_s000
                                 + 0.869985 w_k1_MAT_s000
                                 + 1.07631 w_k1_MSFT_s000
                                 + 0.931761 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.844941 w_k1_ADM_s000
                                 + 1.07404 w_k1_AXP_s000
                                 + 1.61033 w_k1_DIS_s000
                                 + 1.14767 w_k1_FDX_s000 + 1.2946 w_k1_IBM_s000
                                 + 0.954058 w_k1_INTC_s000
                                 + 1.35778 w_k1_KO_s000 + 1.44304 w_k1_MAT_s000
                                 + 1.13883 w_k1_MSFT_s000
                                 + 1.19587 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.907904 w_k1_ADM_s000
                                 + 0.873768 w_k1_AXP_s000
                                 + 0.76919 w_k1_DIS_s000
                                 + 0.914191 w_k1_FDX_s000
                                 + 0.721108 w_k1_IBM_s000
                                 + 0.768718 w_k1_INTC_s000
                                 + 0.789402 w_k1_KO_s000
                                 + 0.853491 w_k1_MAT_s000
                                 + 0.987194 w_k1_MSFT_s000
                                 + 0.897359 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.978077 w_k1_ADM_s001
                                 + 0.762122 w_k1_AXP_s001
                                 + 1.30888 w_k1_DIS_s001
                                 + 0.905018 w_k1_FDX_s001
                                 + 1.06782 w_k1_IBM_s001
                                 + 1.36887 w_k1_INTC_s001
                                 + 1.16414 w_k1_KO_s001
                                 + 0.539959 w_k1_MAT_s001
                                 + 1.12079 w_k1_MSFT_s001
                                 + 0.961251 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.838931 w_k1_ADM_s001
                                 + 0.785652 w_k1_AXP_s001
                                 + 0.761465 w_k1_DIS_s001
                                 + 0.845641 w_k1_FDX_s001
                                 + 0.819419 w_k1_IBM_s001
                                 + 0.980966 w_k1_INTC_s001
                                 + 0.82259 w_k1_KO_s001
                                 + 0.870118 w_k1_MAT_s001
                                 + 0.83774 w_k1_MSFT_s001
                                 + 0.881324 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.835959 w_k1_ADM_s001
                                 + 0.948775 w_k1_AXP_s001
                                 + 0.862838 w_k1_DIS_s001
                                 + 0.857209 w_k1_FDX_s001
                                 + 1.00865 w_k1_IBM_s001
                                 + 0.995295 w_k1_INTC_s001
                                 + 1.08278 w_k1_KO_s001 + 1.15934 w_k1_MAT_s001
                                 + 0.876886 w_k1_MSFT_s001
                                 + 1.03295 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.999647 w_k1_ADM_s001
                                 + 1.24122 w_k1_AXP_s001
                                 + 1.22362 w_k1_DIS_s001
                                 + 1.14114 w_k1_FDX_s001
                                 + 1.16028 w_k1_IBM_s001
                                 + 1.24162 w_k1_INTC_s001 + 1.1312 w_k1_KO_s001
                                 + 1.03017 w_k1_MAT_s001
                                 + 1.18586 w_k1_MSFT_s001
                                 + 0.96275 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.19241 w_k1_ADM_s001
                                 + 1.48373 w_k1_AXP_s001
                                 + 1.33469 w_k1_DIS_s001
                                 + 1.38478 w_k1_FDX_s001
                                 + 1.09173 w_k1_IBM_s001
                                 + 1.44478 w_k1_INTC_s001
                                 + 1.11903 w_k1_KO_s001 + 1.22839 w_k1_MAT_s001
                                 + 1.24202 w_k1_MSFT_s001
                                 + 1.13133 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.4726 w_k1_ADM_s001
                                 + 1.33724 w_k1_AXP_s001
                                 + 1.61699 w_k1_DIS_s001
                                 + 1.28764 w_k1_FDX_s001
                                 + 1.53888 w_k1_IBM_s001
                                 + 1.61586 w_k1_INTC_s001
                                 + 1.40172 w_k1_KO_s001 + 1.36107 w_k1_MAT_s001
                                 + 1.3866 w_k1_MSFT_s001 + 1.23781 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.96021 w_k1_ADM_s001
                                 + 1.14148 w_k1_AXP_s001
                                 + 0.925919 w_k1_DIS_s001
                                 + 0.919017 w_k1_FDX_s001
                                 + 1.30238 w_k1_IBM_s001 + 1.027 w_k1_INTC_s001
                                 + 0.975647 w_k1_KO_s001 + 1.071 w_k1_MAT_s001
                                 + 0.949749 w_k1_MSFT_s001
                                 + 1.01837 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.879821 w_k1_ADM_s001
                                 + 0.974028 w_k1_AXP_s001
                                 + 1.11228 w_k1_DIS_s001
                                 + 1.07195 w_k1_FDX_s001
                                 + 0.949365 w_k1_IBM_s001
                                 + 1.30811 w_k1_INTC_s001
                                 + 0.993984 w_k1_KO_s001
                                 + 0.926091 w_k1_MAT_s001
                                 + 1.10698 w_k1_MSFT_s001
                                 + 1.03663 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.14053 w_k1_ADM_s002
                                 + 0.660729 w_k1_AXP_s002
                                 + 0.780904 w_k1_DIS_s002
                                 + 0.840973 w_k1_FDX_s002
                                 + 0.873294 w_k1_IBM_s002
                                 + 0.988295 w_k1_INTC_s002
                                 + 0.884318 w_k1_KO_s002
                                 + 0.632973 w_k1_MAT_s002
                                 + 0.931733 w_k1_MSFT_s002
                                 + 0.90718 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.36288 w_k1_ADM_s002
                                 + 1.0967 w_k1_AXP_s002 + 1.11205 w_k1_DIS_s002
                                 + 1.00532 w_k1_FDX_s002
                                 + 0.792436 w_k1_IBM_s002
                                 + 1.11251 w_k1_INTC_s002
                                 + 1.05965 w_k1_KO_s002
                                 + 0.687009 w_k1_MAT_s002
                                 + 1.11535 w_k1_MSFT_s002
                                 + 1.15262 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.43564 w_k1_ADM_s002
                                 + 1.11903 w_k1_AXP_s002
                                 + 1.33093 w_k1_DIS_s002
                                 + 1.35901 w_k1_FDX_s002
                                 + 0.72051 w_k1_IBM_s002
                                 + 1.27277 w_k1_INTC_s002
                                 + 1.07882 w_k1_KO_s002
                                 + 0.648832 w_k1_MAT_s002
                                 + 1.29844 w_k1_MSFT_s002
                                 + 1.13021 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.992928 w_k1_ADM_s002
                                 + 0.70374 w_k1_AXP_s002
                                 + 0.663388 w_k1_DIS_s002
                                 + 0.817862 w_k1_FDX_s002
                                 + 0.584233 w_k1_IBM_s002
                                 + 0.975405 w_k1_INTC_s002
                                 + 0.649439 w_k1_KO_s002
                                 + 0.430318 w_k1_MAT_s002
                                 + 0.809068 w_k1_MSFT_s002
                                 + 0.739333 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.31361 w_k1_ADM_s002
                                 + 0.830993 w_k1_AXP_s002
                                 + 0.882021 w_k1_DIS_s002
                                 + 0.861701 w_k1_FDX_s002
                                 + 1.21732 w_k1_IBM_s002
                                 + 0.885992 w_k1_INTC_s002
                                 + 1.04123 w_k1_KO_s002
                                 + 0.967556 w_k1_MAT_s002
                                 + 0.978888 w_k1_MSFT_s002
                                 + 0.879004 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.40595 w_k1_ADM_s002
                                 + 1.65767 w_k1_AXP_s002
                                 + 1.15739 w_k1_DIS_s002
                                 + 1.34329 w_k1_FDX_s002
                                 + 0.860662 w_k1_IBM_s002
                                 + 1.3135 w_k1_INTC_s002 + 1.11693 w_k1_KO_s002
                                 + 1.20318 w_k1_MAT_s002
                                 + 1.2921 w_k1_MSFT_s002 + 1.15253 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.44562 w_k1_ADM_s002
                                 + 1.15274 w_k1_AXP_s002
                                 + 1.47829 w_k1_DIS_s002
                                 + 1.14584 w_k1_FDX_s002
                                 + 1.07795 w_k1_IBM_s002
                                 + 1.39341 w_k1_INTC_s002
                                 + 1.36867 w_k1_KO_s002 + 1.02346 w_k1_MAT_s002
                                 + 1.27684 w_k1_MSFT_s002
                                 + 1.15163 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.22953 w_k1_ADM_s002
                                 + 1.55839 w_k1_AXP_s002
                                 + 1.48626 w_k1_DIS_s002
                                 + 1.31308 w_k1_FDX_s002 + 1.6343 w_k1_IBM_s002
                                 + 1.41508 w_k1_INTC_s002
                                 + 1.57309 w_k1_KO_s002 + 1.84189 w_k1_MAT_s002
                                 + 1.3168 w_k1_MSFT_s002 + 1.33433 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.49216 w_k1_ADM_s003
                                 + 1.54574 w_k1_AXP_s003
                                 + 1.27148 w_k1_DIS_s003
                                 + 1.22841 w_k1_FDX_s003
                                 + 1.07101 w_k1_IBM_s003
                                 + 0.923181 w_k1_INTC_s003
                                 + 1.17334 w_k1_KO_s003 + 1.44008 w_k1_MAT_s003
                                 + 1.03094 w_k1_MSFT_s003
                                 + 1.13517 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.874114 w_k1_ADM_s003
                                 + 0.662147 w_k1_AXP_s003
                                 + 1.04829 w_k1_DIS_s003
                                 + 0.873187 w_k1_FDX_s003
                                 + 1.04182 w_k1_IBM_s003
                                 + 0.889247 w_k1_INTC_s003
                                 + 1.05045 w_k1_KO_s003 + 0.84459 w_k1_MAT_s003
                                 + 1.06671 w_k1_MSFT_s003
                                 + 0.939558 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.32941 w_k1_ADM_s003
                                 + 1.07684 w_k1_AXP_s003
                                 + 1.53447 w_k1_DIS_s003
                                 + 1.28563 w_k1_FDX_s003
                                 + 0.993787 w_k1_IBM_s003
                                 + 1.01176 w_k1_INTC_s003
                                 + 1.35777 w_k1_KO_s003
                                 + 0.912156 w_k1_MAT_s003
                                 + 1.21459 w_k1_MSFT_s003
                                 + 1.19035 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.03455 w_k1_ADM_s003
                                 + 1.03422 w_k1_AXP_s003
                                 + 0.910564 w_k1_DIS_s003
                                 + 0.988896 w_k1_FDX_s003
                                 + 0.638189 w_k1_IBM_s003
                                 + 0.887645 w_k1_INTC_s003
                                 + 0.795007 w_k1_KO_s003
                                 + 0.578605 w_k1_MAT_s003
                                 + 1.00347 w_k1_MSFT_s003
                                 + 0.992192 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.34476 w_k1_ADM_s003
                                 + 1.25206 w_k1_AXP_s003
                                 + 1.36215 w_k1_DIS_s003 + 1.2221 w_k1_FDX_s003
                                 + 1.44908 w_k1_IBM_s003
                                 + 0.941939 w_k1_INTC_s003
                                 + 1.4334 w_k1_KO_s003 + 1.34967 w_k1_MAT_s003
                                 + 1.21118 w_k1_MSFT_s003
                                 + 1.15113 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.07798 w_k1_ADM_s003
                                 + 0.897138 w_k1_AXP_s003
                                 + 0.827609 w_k1_DIS_s003
                                 + 0.880798 w_k1_FDX_s003
                                 + 1.62549 w_k1_IBM_s003
                                 + 0.777158 w_k1_INTC_s003
                                 + 1.20281 w_k1_KO_s003 + 1.45334 w_k1_MAT_s003
                                 + 0.855716 w_k1_MSFT_s003
                                 + 0.960385 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.08456 w_k1_ADM_s003
                                 + 1.13631 w_k1_AXP_s003
                                 + 0.82522 w_k1_DIS_s003
                                 + 0.930176 w_k1_FDX_s003
                                 + 0.965094 w_k1_IBM_s003
                                 + 0.884577 w_k1_INTC_s003
                                 + 0.90691 w_k1_KO_s003
                                 + 0.891977 w_k1_MAT_s003
                                 + 0.968123 w_k1_MSFT_s003
                                 + 0.913773 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.630282 w_k1_ADM_s003
                                 + 0.644442 w_k1_AXP_s003 + 0.582 w_k1_DIS_s003
                                 + 0.641102 w_k1_FDX_s003
                                 + 0.740088 w_k1_IBM_s003
                                 + 0.750178 w_k1_INTC_s003
                                 + 0.742026 w_k1_KO_s003
                                 + 0.65957 w_k1_MAT_s003
                                 + 0.78889 w_k1_MSFT_s003
                                 + 0.796908 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.808599 w_k1_ADM_s004
                                 + 0.764138 w_k1_AXP_s004
                                 + 0.883658 w_k1_DIS_s004
                                 + 0.893605 w_k1_FDX_s004
                                 + 0.837027 w_k1_IBM_s004
                                 + 0.976002 w_k1_INTC_s004
                                 + 1.00782 w_k1_KO_s004
                                 + 0.500886 w_k1_MAT_s004
                                 + 1.0012 w_k1_MSFT_s004
                                 + 0.913822 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.77983 w_k1_ADM_s004
                                 + 0.648716 w_k1_AXP_s004
                                 + 0.630042 w_k1_DIS_s004
                                 + 0.763309 w_k1_FDX_s004
                                 + 0.797167 w_k1_IBM_s004
                                 + 0.946083 w_k1_INTC_s004
                                 + 0.815124 w_k1_KO_s004
                                 + 0.789842 w_k1_MAT_s004
                                 + 0.90874 w_k1_MSFT_s004
                                 + 0.963736 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.38631 w_k1_ADM_s004
                                 + 1.43554 w_k1_AXP_s004
                                 + 1.55782 w_k1_DIS_s004 + 1.3642 w_k1_FDX_s004
                                 + 1.35771 w_k1_IBM_s004
                                 + 1.17843 w_k1_INTC_s004
                                 + 1.51608 w_k1_KO_s004 + 1.56475 w_k1_MAT_s004
                                 + 1.29236 w_k1_MSFT_s004
                                 + 1.51236 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.99085 w_k1_ADM_s004
                                 + 0.965879 w_k1_AXP_s004
                                 + 1.12689 w_k1_DIS_s004
                                 + 1.17025 w_k1_FDX_s004
                                 + 0.801474 w_k1_IBM_s004
                                 + 1.04319 w_k1_INTC_s004
                                 + 0.973354 w_k1_KO_s004 + 0.7852 w_k1_MAT_s004
                                 + 1.28456 w_k1_MSFT_s004
                                 + 1.17006 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.05542 w_k1_ADM_s004
                                 + 1.1117 w_k1_AXP_s004 + 1.26049 w_k1_DIS_s004
                                 + 1.09421 w_k1_FDX_s004
                                 + 1.34762 w_k1_IBM_s004
                                 + 1.07788 w_k1_INTC_s004
                                 + 1.28651 w_k1_KO_s004 + 1.44091 w_k1_MAT_s004
                                 + 1.16448 w_k1_MSFT_s004
                                 + 1.23901 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.95445 w_k1_ADM_s004
                                 + 0.885462 w_k1_AXP_s004
                                 + 0.977157 w_k1_DIS_s004
                                 + 0.908477 w_k1_FDX_s004
                                 + 0.956815 w_k1_IBM_s004
                                 + 0.904382 w_k1_INTC_s004
                                 + 0.941745 w_k1_KO_s004
                                 + 0.951899 w_k1_MAT_s004
                                 + 1.05153 w_k1_MSFT_s004
                                 + 1.23587 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.26892 w_k1_ADM_s004
                                 + 1.22667 w_k1_AXP_s004
                                 + 0.849296 w_k1_DIS_s004
                                 + 1.04068 w_k1_FDX_s004
                                 + 0.870705 w_k1_IBM_s004
                                 + 0.981015 w_k1_INTC_s004
                                 + 0.99658 w_k1_KO_s004
                                 + 0.975408 w_k1_MAT_s004
                                 + 0.954909 w_k1_MSFT_s004
                                 + 1.10813 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.03998 w_k1_ADM_s004
                                 + 0.996238 w_k1_AXP_s004
                                 + 1.01377 w_k1_DIS_s004
                                 + 0.927422 w_k1_FDX_s004
                                 + 1.43209 w_k1_IBM_s004
                                 + 0.808601 w_k1_INTC_s004
                                 + 1.13889 w_k1_KO_s004 + 1.57318 w_k1_MAT_s004
                                 + 0.975744 w_k1_MSFT_s004
                                 + 1.16373 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.28096 w_k1_ADM_s005
                                 + 1.15569 w_k1_AXP_s005
                                 + 0.836367 w_k1_DIS_s005
                                 + 1.10702 w_k1_FDX_s005
                                 + 0.676367 w_k1_IBM_s005
                                 + 0.881698 w_k1_INTC_s005
                                 + 0.781806 w_k1_KO_s005
                                 + 0.938634 w_k1_MAT_s005
                                 + 0.97173 w_k1_MSFT_s005
                                 + 1.01696 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.829085 w_k1_ADM_s005
                                 + 0.824148 w_k1_AXP_s005
                                 + 0.987912 w_k1_DIS_s005
                                 + 0.991225 w_k1_FDX_s005
                                 + 0.774119 w_k1_IBM_s005
                                 + 0.961452 w_k1_INTC_s005 + 0.882 w_k1_KO_s005
                                 + 0.742887 w_k1_MAT_s005
                                 + 1.07661 w_k1_MSFT_s005
                                 + 0.959993 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.02727 w_k1_ADM_s005
                                 + 0.781894 w_k1_AXP_s005
                                 + 0.992836 w_k1_DIS_s005
                                 + 0.966321 w_k1_FDX_s005
                                 + 1.08375 w_k1_IBM_s005
                                 + 0.857236 w_k1_INTC_s005
                                 + 0.965782 w_k1_KO_s005
                                 + 0.696716 w_k1_MAT_s005
                                 + 0.95348 w_k1_MSFT_s005
                                 + 0.885217 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.11989 w_k1_ADM_s005
                                 + 1.85694 w_k1_AXP_s005
                                 + 1.08609 w_k1_DIS_s005
                                 + 1.12322 w_k1_FDX_s005
                                 + 1.07354 w_k1_IBM_s005
                                 + 1.18086 w_k1_INTC_s005
                                 + 1.02342 w_k1_KO_s005 + 1.58856 w_k1_MAT_s005
                                 + 1.05792 w_k1_MSFT_s005
                                 + 1.13879 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.87836 w_k1_ADM_s005
                                 + 0.980047 w_k1_AXP_s005
                                 + 0.860376 w_k1_DIS_s005
                                 + 0.77194 w_k1_FDX_s005
                                 + 1.27145 w_k1_IBM_s005
                                 + 0.641139 w_k1_INTC_s005
                                 + 0.939545 w_k1_KO_s005
                                 + 1.38695 w_k1_MAT_s005
                                 + 0.779141 w_k1_MSFT_s005
                                 + 0.907134 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.768602 w_k1_ADM_s005
                                 + 0.770855 w_k1_AXP_s005
                                 + 0.606534 w_k1_DIS_s005
                                 + 0.787463 w_k1_FDX_s005
                                 + 0.830428 w_k1_IBM_s005
                                 + 0.706965 w_k1_INTC_s005
                                 + 0.724554 w_k1_KO_s005
                                 + 0.784174 w_k1_MAT_s005
                                 + 0.715022 w_k1_MSFT_s005
                                 + 0.813045 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.0963 w_k1_ADM_s005
                                 + 1.48269 w_k1_AXP_s005
                                 + 1.66595 w_k1_DIS_s005
                                 + 1.33721 w_k1_FDX_s005
                                 + 1.29199 w_k1_IBM_s005
                                 + 1.27308 w_k1_INTC_s005
                                 + 1.34504 w_k1_KO_s005 + 1.26159 w_k1_MAT_s005
                                 + 1.26683 w_k1_MSFT_s005
                                 + 1.26502 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.869953 w_k1_ADM_s005
                                 + 1.02992 w_k1_AXP_s005
                                 + 1.12762 w_k1_DIS_s005
                                 + 1.02469 w_k1_FDX_s005
                                 + 1.09514 w_k1_IBM_s005
                                 + 1.19841 w_k1_INTC_s005
                                 + 0.985504 w_k1_KO_s005
                                 + 1.03825 w_k1_MAT_s005
                                 + 1.08104 w_k1_MSFT_s005
                                 + 1.08761 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.910979 w_k1_ADM_s006
                                 + 0.929654 w_k1_AXP_s006
                                 + 0.878107 w_k1_DIS_s006
                                 + 0.918404 w_k1_FDX_s006
                                 + 0.972758 w_k1_IBM_s006
                                 + 0.840398 w_k1_INTC_s006
                                 + 0.820938 w_k1_KO_s006
                                 + 1.18008 w_k1_MAT_s006
                                 + 0.899955 w_k1_MSFT_s006
                                 + 0.88999 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.74481 w_k1_ADM_s006
                                 + 0.716494 w_k1_AXP_s006
                                 + 0.607273 w_k1_DIS_s006
                                 + 0.704273 w_k1_FDX_s006
                                 + 0.842182 w_k1_IBM_s006
                                 + 0.809864 w_k1_INTC_s006
                                 + 0.735002 w_k1_KO_s006
                                 + 1.08259 w_k1_MAT_s006
                                 + 0.797002 w_k1_MSFT_s006
                                 + 0.78278 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.01951 w_k1_ADM_s006
                                 + 0.97232 w_k1_AXP_s006
                                 + 0.894581 w_k1_DIS_s006
                                 + 0.989635 w_k1_FDX_s006
                                 + 0.638261 w_k1_IBM_s006
                                 + 0.923162 w_k1_INTC_s006
                                 + 0.790574 w_k1_KO_s006
                                 + 0.783612 w_k1_MAT_s006
                                 + 0.95727 w_k1_MSFT_s006
                                 + 0.913033 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.981793 w_k1_ADM_s006
                                 + 0.962565 w_k1_AXP_s006
                                 + 0.814125 w_k1_DIS_s006
                                 + 0.784542 w_k1_FDX_s006
                                 + 1.5116 w_k1_IBM_s006
                                 + 0.841785 w_k1_INTC_s006
                                 + 1.13381 w_k1_KO_s006 + 2.18097 w_k1_MAT_s006
                                 + 0.800855 w_k1_MSFT_s006
                                 + 0.929768 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.03204 w_k1_ADM_s006
                                 + 1.14304 w_k1_AXP_s006
                                 + 1.22122 w_k1_DIS_s006
                                 + 1.05565 w_k1_FDX_s006
                                 + 1.30104 w_k1_IBM_s006
                                 + 0.935583 w_k1_INTC_s006
                                 + 0.971415 w_k1_KO_s006
                                 + 1.23532 w_k1_MAT_s006
                                 + 1.13545 w_k1_MSFT_s006
                                 + 1.02926 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.21544 w_k1_ADM_s006
                                 + 1.49801 w_k1_AXP_s006 + 1.2103 w_k1_DIS_s006
                                 + 1.44092 w_k1_FDX_s006
                                 + 0.729113 w_k1_IBM_s006
                                 + 1.14744 w_k1_INTC_s006
                                 + 0.883987 w_k1_KO_s006
                                 + 0.868177 w_k1_MAT_s006
                                 + 1.15109 w_k1_MSFT_s006
                                 + 1.08379 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.06129 w_k1_ADM_s006
                                 + 1.01489 w_k1_AXP_s006
                                 + 1.09826 w_k1_DIS_s006
                                 + 1.08669 w_k1_FDX_s006
                                 + 0.809096 w_k1_IBM_s006
                                 + 0.931257 w_k1_INTC_s006
                                 + 0.892898 w_k1_KO_s006
                                 + 0.853926 w_k1_MAT_s006
                                 + 1.06747 w_k1_MSFT_s006
                                 + 0.897075 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.17026 w_k1_ADM_s006
                                 + 1.56503 w_k1_AXP_s006
                                 + 1.61094 w_k1_DIS_s006
                                 + 1.36947 w_k1_FDX_s006
                                 + 1.38828 w_k1_IBM_s006
                                 + 1.14695 w_k1_INTC_s006
                                 + 1.23982 w_k1_KO_s006 + 2.174 w_k1_MAT_s006
                                 + 1.21745 w_k1_MSFT_s006
                                 + 1.14673 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.07473 w_k1_ADM_s007
                                 + 1.34672 w_k1_AXP_s007
                                 + 1.47528 w_k1_DIS_s007
                                 + 1.34776 w_k1_FDX_s007
                                 + 1.10032 w_k1_IBM_s007
                                 + 1.42366 w_k1_INTC_s007
                                 + 1.43129 w_k1_KO_s007 + 1.06956 w_k1_MAT_s007
                                 + 1.26754 w_k1_MSFT_s007
                                 + 1.18317 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.874073 w_k1_ADM_s007
                                 + 0.793857 w_k1_AXP_s007
                                 + 0.643108 w_k1_DIS_s007
                                 + 0.726864 w_k1_FDX_s007
                                 + 1.08592 w_k1_IBM_s007
                                 + 0.777456 w_k1_INTC_s007
                                 + 1.01374 w_k1_KO_s007 + 1.07463 w_k1_MAT_s007
                                 + 0.758913 w_k1_MSFT_s007
                                 + 0.866712 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.13966 w_k1_ADM_s007
                                 + 1.0053 w_k1_AXP_s007 + 1.21064 w_k1_DIS_s007
                                 + 1.15793 w_k1_FDX_s007
                                 + 0.777946 w_k1_IBM_s007
                                 + 1.23482 w_k1_INTC_s007
                                 + 1.16399 w_k1_KO_s007
                                 + 0.780797 w_k1_MAT_s007
                                 + 1.12138 w_k1_MSFT_s007
                                 + 1.06728 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.986117 w_k1_ADM_s007
                                 + 0.79581 w_k1_AXP_s007
                                 + 0.887075 w_k1_DIS_s007
                                 + 0.895527 w_k1_FDX_s007
                                 + 1.09937 w_k1_IBM_s007
                                 + 0.917533 w_k1_INTC_s007
                                 + 1.15643 w_k1_KO_s007
                                 + 0.793255 w_k1_MAT_s007
                                 + 0.856492 w_k1_MSFT_s007
                                 + 0.970465 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.834445 w_k1_ADM_s007
                                 + 0.734614 w_k1_AXP_s007
                                 + 0.735955 w_k1_DIS_s007
                                 + 0.824746 w_k1_FDX_s007
                                 + 0.653033 w_k1_IBM_s007
                                 + 1.08415 w_k1_INTC_s007
                                 + 0.819318 w_k1_KO_s007
                                 + 0.74331 w_k1_MAT_s007
                                 + 0.90139 w_k1_MSFT_s007
                                 + 0.855713 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.893287 w_k1_ADM_s007
                                 + 1.06198 w_k1_AXP_s007
                                 + 0.912879 w_k1_DIS_s007
                                 + 0.983869 w_k1_FDX_s007
                                 + 1.07163 w_k1_IBM_s007
                                 + 0.994945 w_k1_INTC_s007
                                 + 1.04135 w_k1_KO_s007 + 1.13916 w_k1_MAT_s007
                                 + 0.854583 w_k1_MSFT_s007
                                 + 0.925074 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.794973 w_k1_ADM_s007
                                 + 1.0178 w_k1_AXP_s007 + 1.17938 w_k1_DIS_s007
                                 + 1.00173 w_k1_FDX_s007
                                 + 1.24355 w_k1_IBM_s007
                                 + 1.05994 w_k1_INTC_s007
                                 + 1.29588 w_k1_KO_s007 + 1.04574 w_k1_MAT_s007
                                 + 0.969906 w_k1_MSFT_s007
                                 + 0.992112 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.867488 w_k1_ADM_s007
                                 + 1.32097 w_k1_AXP_s007
                                 + 1.52721 w_k1_DIS_s007
                                 + 1.19937 w_k1_FDX_s007
                                 + 1.63837 w_k1_IBM_s007
                                 + 1.45625 w_k1_INTC_s007
                                 + 1.65656 w_k1_KO_s007 + 1.89533 w_k1_MAT_s007
                                 + 1.17172 w_k1_MSFT_s007
                                 + 1.23451 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.698962 w_k1_ADM_s010
                                 + 0.765554 w_k1_AXP_s010
                                 + 0.707151 w_k1_DIS_s010
                                 + 0.774993 w_k1_FDX_s010
                                 + 0.97952 w_k1_IBM_s010
                                 + 0.932537 w_k1_INTC_s010
                                 + 0.830456 w_k1_KO_s010
                                 + 1.29881 w_k1_MAT_s010
                                 + 0.890741 w_k1_MSFT_s010
                                 + 0.971949 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.799738 w_k1_ADM_s010
                                 + 0.798677 w_k1_AXP_s010
                                 + 1.19559 w_k1_DIS_s010
                                 + 1.00301 w_k1_FDX_s010
                                 + 1.14655 w_k1_IBM_s010
                                 + 1.16224 w_k1_INTC_s010
                                 + 1.21412 w_k1_KO_s010 + 1.80311 w_k1_MAT_s010
                                 + 1.15773 w_k1_MSFT_s010
                                 + 1.06216 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.18088 w_k1_ADM_s010
                                 + 1.1027 w_k1_AXP_s010 + 1.46205 w_k1_DIS_s010
                                 + 1.23641 w_k1_FDX_s010
                                 + 1.03703 w_k1_IBM_s010
                                 + 1.58192 w_k1_INTC_s010 + 1.3286 w_k1_KO_s010
                                 + 1.02785 w_k1_MAT_s010
                                 + 1.35519 w_k1_MSFT_s010 + 1.2002 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.22745 w_k1_ADM_s010
                                 + 1.18449 w_k1_AXP_s010
                                 + 1.30204 w_k1_DIS_s010 + 1.2249 w_k1_FDX_s010
                                 + 1.38226 w_k1_IBM_s010
                                 + 1.35419 w_k1_INTC_s010
                                 + 1.18114 w_k1_KO_s010 + 1.7391 w_k1_MAT_s010
                                 + 1.136 w_k1_MSFT_s010 + 1.14397 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.982779 w_k1_ADM_s010
                                 + 1.10935 w_k1_AXP_s010
                                 + 0.961626 w_k1_DIS_s010
                                 + 0.951926 w_k1_FDX_s010
                                 + 1.56073 w_k1_IBM_s010
                                 + 1.08758 w_k1_INTC_s010 + 1.2662 w_k1_KO_s010
                                 + 1.39988 w_k1_MAT_s010
                                 + 0.970725 w_k1_MSFT_s010
                                 + 0.963432 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.47161 w_k1_ADM_s010
                                 + 1.67544 w_k1_AXP_s010
                                 + 1.64841 w_k1_DIS_s010
                                 + 1.51493 w_k1_FDX_s010
                                 + 1.21566 w_k1_IBM_s010
                                 + 1.36302 w_k1_INTC_s010 + 1.378 w_k1_KO_s010
                                 + 1.90033 w_k1_MAT_s010
                                 + 1.23835 w_k1_MSFT_s010
                                 + 1.38045 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.00128 w_k1_ADM_s010
                                 + 0.864863 w_k1_AXP_s010
                                 + 0.95274 w_k1_DIS_s010
                                 + 0.99352 w_k1_FDX_s010
                                 + 0.808668 w_k1_IBM_s010
                                 + 1.06309 w_k1_INTC_s010
                                 + 1.01294 w_k1_KO_s010
                                 + 0.905297 w_k1_MAT_s010
                                 + 1.05212 w_k1_MSFT_s010
                                 + 0.961681 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.952444 w_k1_ADM_s010
                                 + 1.2652 w_k1_AXP_s010
                                 + 0.961344 w_k1_DIS_s010
                                 + 1.05324 w_k1_FDX_s010 + 1.2066 w_k1_IBM_s010
                                 + 1.26484 w_k1_INTC_s010
                                 + 1.15369 w_k1_KO_s010 + 1.62057 w_k1_MAT_s010
                                 + 1.07515 w_k1_MSFT_s010
                                 + 1.04636 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.679738 w_k1_ADM_s011
                                 + 0.84708 w_k1_AXP_s011
                                 + 1.17184 w_k1_DIS_s011
                                 + 0.898357 w_k1_FDX_s011
                                 + 1.18044 w_k1_IBM_s011
                                 + 1.0203 w_k1_INTC_s011 + 1.26582 w_k1_KO_s011
                                 + 1.06832 w_k1_MAT_s011
                                 + 1.06857 w_k1_MSFT_s011
                                 + 0.997791 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.41023 w_k1_ADM_s011
                                 + 1.27512 w_k1_AXP_s011
                                 + 1.44066 w_k1_DIS_s011
                                 + 1.36883 w_k1_FDX_s011
                                 + 1.06972 w_k1_IBM_s011
                                 + 1.21592 w_k1_INTC_s011
                                 + 1.22066 w_k1_KO_s011 + 1.04633 w_k1_MAT_s011
                                 + 1.16143 w_k1_MSFT_s011
                                 + 1.10847 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.924226 w_k1_ADM_s011
                                 + 1.00419 w_k1_AXP_s011
                                 + 0.784787 w_k1_DIS_s011
                                 + 0.918936 w_k1_FDX_s011
                                 + 0.941916 w_k1_IBM_s011
                                 + 0.881508 w_k1_INTC_s011
                                 + 0.938664 w_k1_KO_s011
                                 + 1.15386 w_k1_MAT_s011
                                 + 0.845725 w_k1_MSFT_s011
                                 + 0.964634 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.09364 w_k1_ADM_s011
                                 + 1.1799 w_k1_AXP_s011 + 1.2616 w_k1_DIS_s011
                                 + 1.1056 w_k1_FDX_s011 + 1.39077 w_k1_IBM_s011
                                 + 0.97098 w_k1_INTC_s011
                                 + 1.22402 w_k1_KO_s011 + 1.58259 w_k1_MAT_s011
                                 + 0.954286 w_k1_MSFT_s011
                                 + 1.07435 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.56326 w_k1_ADM_s011
                                 + 0.797515 w_k1_AXP_s011
                                 + 0.790881 w_k1_DIS_s011
                                 + 0.738535 w_k1_FDX_s011
                                 + 1.04194 w_k1_IBM_s011
                                 + 0.768778 w_k1_INTC_s011
                                 + 0.98969 w_k1_KO_s011
                                 + 0.995787 w_k1_MAT_s011
                                 + 0.860723 w_k1_MSFT_s011
                                 + 0.845524 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.03087 w_k1_ADM_s011
                                 + 1.23468 w_k1_AXP_s011
                                 + 1.87069 w_k1_DIS_s011
                                 + 1.27895 w_k1_FDX_s011
                                 + 1.32283 w_k1_IBM_s011
                                 + 1.36117 w_k1_INTC_s011
                                 + 1.81683 w_k1_KO_s011 + 1.4905 w_k1_MAT_s011
                                 + 1.35631 w_k1_MSFT_s011
                                 + 1.12408 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.09995 w_k1_ADM_s011
                                 + 1.04179 w_k1_AXP_s011
                                 + 0.884929 w_k1_DIS_s011
                                 + 1.12006 w_k1_FDX_s011
                                 + 0.661282 w_k1_IBM_s011
                                 + 0.993082 w_k1_INTC_s011
                                 + 0.784093 w_k1_KO_s011
                                 + 0.759958 w_k1_MAT_s011
                                 + 0.927376 w_k1_MSFT_s011
                                 + 0.962547 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.946555 w_k1_ADM_s011
                                 + 1.04472 w_k1_AXP_s011
                                 + 1.09764 w_k1_DIS_s011
                                 + 0.961072 w_k1_FDX_s011
                                 + 1.00566 w_k1_IBM_s011
                                 + 0.992504 w_k1_INTC_s011
                                 + 0.979618 w_k1_KO_s011
                                 + 0.825537 w_k1_MAT_s011
                                 + 0.861902 w_k1_MSFT_s011
                                 + 0.939858 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.835767 w_k1_ADM_s012
                                 + 0.797707 w_k1_AXP_s012
                                 + 0.648898 w_k1_DIS_s012
                                 + 0.851924 w_k1_FDX_s012
                                 + 0.814227 w_k1_IBM_s012
                                 + 0.854323 w_k1_INTC_s012
                                 + 0.765741 w_k1_KO_s012
                                 + 0.798761 w_k1_MAT_s012
                                 + 0.727138 w_k1_MSFT_s012
                                 + 0.939769 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.16045 w_k1_ADM_s012
                                 + 1.08756 w_k1_AXP_s012
                                 + 1.15238 w_k1_DIS_s012
                                 + 1.02195 w_k1_FDX_s012 + 1.4709 w_k1_IBM_s012
                                 + 0.853192 w_k1_INTC_s012
                                 + 1.18216 w_k1_KO_s012 + 1.65746 w_k1_MAT_s012
                                 + 0.879123 w_k1_MSFT_s012
                                 + 1.09075 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.78551 w_k1_ADM_s012
                                 + 1.50438 w_k1_AXP_s012
                                 + 1.11707 w_k1_DIS_s012
                                 + 1.06322 w_k1_FDX_s012
                                 + 1.47417 w_k1_IBM_s012 + 1.017 w_k1_INTC_s012
                                 + 1.33316 w_k1_KO_s012 + 2.05343 w_k1_MAT_s012
                                 + 0.917474 w_k1_MSFT_s012
                                 + 1.33969 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01365 w_k1_ADM_s012
                                 + 1.09435 w_k1_AXP_s012
                                 + 1.04319 w_k1_DIS_s012
                                 + 1.00686 w_k1_FDX_s012
                                 + 1.07134 w_k1_IBM_s012
                                 + 0.909506 w_k1_INTC_s012
                                 + 1.02856 w_k1_KO_s012 + 1.34024 w_k1_MAT_s012
                                 + 0.802121 w_k1_MSFT_s012
                                 + 1.31737 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.87773 w_k1_ADM_s012
                                 + 0.745913 w_k1_AXP_s012
                                 + 1.02907 w_k1_DIS_s012
                                 + 0.872791 w_k1_FDX_s012
                                 + 0.960742 w_k1_IBM_s012
                                 + 0.869553 w_k1_INTC_s012
                                 + 1.00994 w_k1_KO_s012 + 0.95818 w_k1_MAT_s012
                                 + 0.77755 w_k1_MSFT_s012
                                 + 1.00921 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.852654 w_k1_ADM_s012
                                 + 0.675505 w_k1_AXP_s012
                                 + 0.615219 w_k1_DIS_s012
                                 + 0.886622 w_k1_FDX_s012
                                 + 0.580202 w_k1_IBM_s012
                                 + 0.712374 w_k1_INTC_s012
                                 + 0.637431 w_k1_KO_s012
                                 + 0.517814 w_k1_MAT_s012
                                 + 0.757334 w_k1_MSFT_s012
                                 + 0.977135 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.993597 w_k1_ADM_s012
                                 + 1.07201 w_k1_AXP_s012
                                 + 1.01556 w_k1_DIS_s012 + 1.1575 w_k1_FDX_s012
                                 + 0.747968 w_k1_IBM_s012
                                 + 1.00766 w_k1_INTC_s012
                                 + 0.925331 w_k1_KO_s012
                                 + 0.939396 w_k1_MAT_s012
                                 + 0.955167 w_k1_MSFT_s012
                                 + 1.16042 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.04832 w_k1_ADM_s012
                                 + 1.20083 w_k1_AXP_s012
                                 + 1.31599 w_k1_DIS_s012
                                 + 1.30045 w_k1_FDX_s012
                                 + 1.07785 w_k1_IBM_s012
                                 + 1.06166 w_k1_INTC_s012
                                 + 1.26531 w_k1_KO_s012 + 1.74811 w_k1_MAT_s012
                                 + 1.11371 w_k1_MSFT_s012
                                 + 1.34778 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.00465 w_k1_ADM_s013
                                 + 1.48603 w_k1_AXP_s013
                                 + 1.34331 w_k1_DIS_s013
                                 + 1.18762 w_k1_FDX_s013
                                 + 1.67869 w_k1_IBM_s013
                                 + 1.35657 w_k1_INTC_s013
                                 + 1.34285 w_k1_KO_s013 + 1.7871 w_k1_MAT_s013
                                 + 1.39098 w_k1_MSFT_s013
                                 + 1.16993 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.32437 w_k1_ADM_s013
                                 + 1.47346 w_k1_AXP_s013 + 1.3762 w_k1_DIS_s013
                                 + 1.28995 w_k1_FDX_s013
                                 + 1.00077 w_k1_IBM_s013
                                 + 1.05963 w_k1_INTC_s013
                                 + 1.05744 w_k1_KO_s013 + 1.615 w_k1_MAT_s013
                                 + 1.2222 w_k1_MSFT_s013 + 1.11781 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.68596 w_k1_ADM_s013
                                 + 0.840602 w_k1_AXP_s013
                                 + 0.818397 w_k1_DIS_s013
                                 + 0.850401 w_k1_FDX_s013
                                 + 0.760344 w_k1_IBM_s013
                                 + 0.793166 w_k1_INTC_s013
                                 + 0.716694 w_k1_KO_s013
                                 + 0.773594 w_k1_MAT_s013
                                 + 0.901713 w_k1_MSFT_s013
                                 + 0.830349 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.13438 w_k1_ADM_s013
                                 + 1.28105 w_k1_AXP_s013
                                 + 1.14636 w_k1_DIS_s013 + 1.2432 w_k1_FDX_s013
                                 + 0.77029 w_k1_IBM_s013
                                 + 1.16617 w_k1_INTC_s013
                                 + 0.870771 w_k1_KO_s013
                                 + 0.967354 w_k1_MAT_s013
                                 + 1.09709 w_k1_MSFT_s013
                                 + 1.04837 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.95426 w_k1_ADM_s013
                                 + 0.904168 w_k1_AXP_s013
                                 + 0.908444 w_k1_DIS_s013
                                 + 0.910166 w_k1_FDX_s013
                                 + 1.11558 w_k1_IBM_s013
                                 + 0.959844 w_k1_INTC_s013
                                 + 0.996543 w_k1_KO_s013
                                 + 1.23918 w_k1_MAT_s013
                                 + 0.998152 w_k1_MSFT_s013
                                 + 0.954454 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.965596 w_k1_ADM_s013
                                 + 0.841206 w_k1_AXP_s013
                                 + 1.32316 w_k1_DIS_s013
                                 + 0.984618 w_k1_FDX_s013
                                 + 1.25343 w_k1_IBM_s013
                                 + 0.843357 w_k1_INTC_s013
                                 + 1.16187 w_k1_KO_s013 + 1.03526 w_k1_MAT_s013
                                 + 1.084 w_k1_MSFT_s013 + 1.05737 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.931071 w_k1_ADM_s013
                                 + 1.23774 w_k1_AXP_s013
                                 + 0.84711 w_k1_DIS_s013
                                 + 1.12374 w_k1_FDX_s013
                                 + 0.729044 w_k1_IBM_s013
                                 + 0.751419 w_k1_INTC_s013
                                 + 0.749429 w_k1_KO_s013
                                 + 1.06395 w_k1_MAT_s013
                                 + 0.864683 w_k1_MSFT_s013
                                 + 0.933449 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.11285 w_k1_ADM_s013
                                 + 1.19012 w_k1_AXP_s013
                                 + 0.864306 w_k1_DIS_s013
                                 + 1.04391 w_k1_FDX_s013
                                 + 1.26324 w_k1_IBM_s013
                                 + 0.93607 w_k1_INTC_s013
                                 + 1.04626 w_k1_KO_s013 + 1.39515 w_k1_MAT_s013
                                 + 1.06908 w_k1_MSFT_s013
                                 + 1.14372 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.870929 w_k1_ADM_s014
                                 + 0.851684 w_k1_AXP_s014
                                 + 1.16945 w_k1_DIS_s014
                                 + 0.98345 w_k1_FDX_s014
                                 + 0.978434 w_k1_IBM_s014
                                 + 1.28539 w_k1_INTC_s014
                                 + 1.07408 w_k1_KO_s014 + 1.00257 w_k1_MAT_s014
                                 + 1.14817 w_k1_MSFT_s014
                                 + 1.12278 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.938415 w_k1_ADM_s014
                                 + 1.05655 w_k1_AXP_s014
                                 + 1.23359 w_k1_DIS_s014
                                 + 0.999587 w_k1_FDX_s014
                                 + 1.64139 w_k1_IBM_s014
                                 + 1.32238 w_k1_INTC_s014
                                 + 1.18945 w_k1_KO_s014 + 1.55708 w_k1_MAT_s014
                                 + 1.0072 w_k1_MSFT_s014 + 1.11084 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.595068 w_k1_ADM_s014
                                 + 0.522063 w_k1_AXP_s014
                                 + 0.505186 w_k1_DIS_s014
                                 + 0.74164 w_k1_FDX_s014
                                 + 0.644823 w_k1_IBM_s014
                                 + 1.05168 w_k1_INTC_s014
                                 + 0.569981 w_k1_KO_s014
                                 + 0.411146 w_k1_MAT_s014
                                 + 0.78823 w_k1_MSFT_s014
                                 + 0.874203 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.01289 w_k1_ADM_s014
                                 + 0.797759 w_k1_AXP_s014
                                 + 0.881014 w_k1_DIS_s014
                                 + 0.971061 w_k1_FDX_s014
                                 + 0.621558 w_k1_IBM_s014
                                 + 1.11927 w_k1_INTC_s014
                                 + 0.753763 w_k1_KO_s014
                                 + 0.448049 w_k1_MAT_s014
                                 + 0.976202 w_k1_MSFT_s014
                                 + 0.978302 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.806915 w_k1_ADM_s014
                                 + 0.803313 w_k1_AXP_s014
                                 + 0.762726 w_k1_DIS_s014
                                 + 0.821269 w_k1_FDX_s014
                                 + 1.23023 w_k1_IBM_s014
                                 + 1.09398 w_k1_INTC_s014
                                 + 0.959808 w_k1_KO_s014
                                 + 0.811952 w_k1_MAT_s014
                                 + 0.861859 w_k1_MSFT_s014
                                 + 0.988291 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.45326 w_k1_ADM_s014
                                 + 1.24242 w_k1_AXP_s014
                                 + 1.09084 w_k1_DIS_s014
                                 + 1.14802 w_k1_FDX_s014
                                 + 1.06292 w_k1_IBM_s014
                                 + 1.27023 w_k1_INTC_s014
                                 + 1.01099 w_k1_KO_s014
                                 + 0.825924 w_k1_MAT_s014
                                 + 1.1626 w_k1_MSFT_s014 + 1.20277 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.49527 w_k1_ADM_s014
                                 + 1.34194 w_k1_AXP_s014
                                 + 1.36576 w_k1_DIS_s014
                                 + 1.42202 w_k1_FDX_s014 + 1.1915 w_k1_IBM_s014
                                 + 1.30675 w_k1_INTC_s014
                                 + 1.31071 w_k1_KO_s014 + 1.4723 w_k1_MAT_s014
                                 + 1.32086 w_k1_MSFT_s014
                                 + 1.30275 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.972481 w_k1_ADM_s014
                                 + 1.02393 w_k1_AXP_s014
                                 + 0.827187 w_k1_DIS_s014
                                 + 0.936918 w_k1_FDX_s014
                                 + 0.916513 w_k1_IBM_s014
                                 + 1.26768 w_k1_INTC_s014
                                 + 0.893367 w_k1_KO_s014
                                 + 0.632442 w_k1_MAT_s014
                                 + 0.881993 w_k1_MSFT_s014
                                 + 0.945428 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.80825 w_k1_ADM_s015
                                 + 0.76083 w_k1_AXP_s015
                                 + 0.631209 w_k1_DIS_s015
                                 + 0.76734 w_k1_FDX_s015
                                 + 0.966934 w_k1_IBM_s015
                                 + 0.77011 w_k1_INTC_s015
                                 + 0.851721 w_k1_KO_s015
                                 + 0.776941 w_k1_MAT_s015
                                 + 0.837484 w_k1_MSFT_s015
                                 + 0.897274 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.928901 w_k1_ADM_s015
                                 + 0.915317 w_k1_AXP_s015
                                 + 1.02366 w_k1_DIS_s015
                                 + 0.904735 w_k1_FDX_s015
                                 + 1.27277 w_k1_IBM_s015
                                 + 0.849627 w_k1_INTC_s015
                                 + 1.18393 w_k1_KO_s015 + 1.10256 w_k1_MAT_s015
                                 + 0.882896 w_k1_MSFT_s015
                                 + 1.00036 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.828567 w_k1_ADM_s015
                                 + 0.768843 w_k1_AXP_s015
                                 + 0.655584 w_k1_DIS_s015
                                 + 0.813509 w_k1_FDX_s015
                                 + 0.715699 w_k1_IBM_s015
                                 + 0.80224 w_k1_INTC_s015
                                 + 0.788661 w_k1_KO_s015
                                 + 0.644757 w_k1_MAT_s015
                                 + 0.831173 w_k1_MSFT_s015
                                 + 0.911163 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.27425 w_k1_ADM_s015
                                 + 1.51416 w_k1_AXP_s015
                                 + 1.54659 w_k1_DIS_s015
                                 + 1.31351 w_k1_FDX_s015
                                 + 1.26657 w_k1_IBM_s015
                                 + 1.36399 w_k1_INTC_s015 + 1.3194 w_k1_KO_s015
                                 + 1.22968 w_k1_MAT_s015
                                 + 1.33256 w_k1_MSFT_s015
                                 + 1.20958 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.925525 w_k1_ADM_s015
                                 + 1.22854 w_k1_AXP_s015
                                 + 1.02756 w_k1_DIS_s015
                                 + 0.932115 w_k1_FDX_s015
                                 + 1.63882 w_k1_IBM_s015
                                 + 0.991699 w_k1_INTC_s015
                                 + 1.4828 w_k1_KO_s015 + 2.06281 w_k1_MAT_s015
                                 + 0.908533 w_k1_MSFT_s015
                                 + 1.15777 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.931332 w_k1_ADM_s015
                                 + 0.998934 w_k1_AXP_s015
                                 + 0.913254 w_k1_DIS_s015
                                 + 0.870725 w_k1_FDX_s015
                                 + 0.895924 w_k1_IBM_s015
                                 + 0.833665 w_k1_INTC_s015
                                 + 0.931502 w_k1_KO_s015
                                 + 0.869947 w_k1_MAT_s015
                                 + 0.925878 w_k1_MSFT_s015
                                 + 1.00197 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.20998 w_k1_ADM_s015
                                 + 1.36558 w_k1_AXP_s015
                                 + 0.895688 w_k1_DIS_s015
                                 + 1.04491 w_k1_FDX_s015
                                 + 1.00403 w_k1_IBM_s015
                                 + 0.994856 w_k1_INTC_s015
                                 + 0.942642 w_k1_KO_s015
                                 + 1.25297 w_k1_MAT_s015
                                 + 0.894918 w_k1_MSFT_s015
                                 + 1.04763 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.33723 w_k1_ADM_s015
                                 + 1.07455 w_k1_AXP_s015
                                 + 1.20054 w_k1_DIS_s015
                                 + 1.18139 w_k1_FDX_s015
                                 + 0.899352 w_k1_IBM_s015
                                 + 1.0838 w_k1_INTC_s015
                                 + 0.906081 w_k1_KO_s015
                                 + 0.843716 w_k1_MAT_s015
                                 + 1.13535 w_k1_MSFT_s015
                                 + 1.05671 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.882475 w_k1_ADM_s016
                                 + 0.741919 w_k1_AXP_s016
                                 + 0.8716 w_k1_DIS_s016
                                 + 0.773512 w_k1_FDX_s016
                                 + 1.14161 w_k1_IBM_s016
                                 + 1.0216 w_k1_INTC_s016
                                 + 0.973119 w_k1_KO_s016
                                 + 0.829873 w_k1_MAT_s016
                                 + 0.884592 w_k1_MSFT_s016
                                 + 0.994958 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.716267 w_k1_ADM_s016
                                 + 0.610974 w_k1_AXP_s016
                                 + 0.762055 w_k1_DIS_s016
                                 + 0.797412 w_k1_FDX_s016
                                 + 0.759425 w_k1_IBM_s016
                                 + 0.947352 w_k1_INTC_s016
                                 + 0.864126 w_k1_KO_s016
                                 + 0.603131 w_k1_MAT_s016
                                 + 0.747661 w_k1_MSFT_s016
                                 + 0.817805 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.16868 w_k1_ADM_s016
                                 + 1.01867 w_k1_AXP_s016
                                 + 0.930574 w_k1_DIS_s016
                                 + 0.961369 w_k1_FDX_s016
                                 + 1.04892 w_k1_IBM_s016
                                 + 0.843998 w_k1_INTC_s016
                                 + 0.908144 w_k1_KO_s016
                                 + 0.808906 w_k1_MAT_s016
                                 + 0.83969 w_k1_MSFT_s016
                                 + 0.923313 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.989805 w_k1_ADM_s016
                                 + 1.04834 w_k1_AXP_s016
                                 + 0.858239 w_k1_DIS_s016
                                 + 0.902158 w_k1_FDX_s016
                                 + 0.725991 w_k1_IBM_s016
                                 + 1.07706 w_k1_INTC_s016
                                 + 0.85475 w_k1_KO_s016
                                 + 0.850563 w_k1_MAT_s016
                                 + 0.919092 w_k1_MSFT_s016
                                 + 0.912698 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.74074 w_k1_ADM_s016
                                 + 0.918039 w_k1_AXP_s016
                                 + 1.01449 w_k1_DIS_s016
                                 + 0.935711 w_k1_FDX_s016
                                 + 1.72676 w_k1_IBM_s016
                                 + 1.04333 w_k1_INTC_s016
                                 + 1.14133 w_k1_KO_s016 + 1.49495 w_k1_MAT_s016
                                 + 0.956399 w_k1_MSFT_s016
                                 + 1.02869 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.985586 w_k1_ADM_s016
                                 + 1.23557 w_k1_AXP_s016
                                 + 1.20802 w_k1_DIS_s016
                                 + 1.03472 w_k1_FDX_s016
                                 + 1.26992 w_k1_IBM_s016
                                 + 1.27192 w_k1_INTC_s016
                                 + 1.15995 w_k1_KO_s016 + 1.36383 w_k1_MAT_s016
                                 + 0.986905 w_k1_MSFT_s016
                                 + 0.975359 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.03226 w_k1_ADM_s016
                                 + 1.19488 w_k1_AXP_s016
                                 + 1.46427 w_k1_DIS_s016
                                 + 1.26452 w_k1_FDX_s016
                                 + 0.96612 w_k1_IBM_s016
                                 + 1.22416 w_k1_INTC_s016
                                 + 1.15938 w_k1_KO_s016 + 0.80586 w_k1_MAT_s016
                                 + 1.1745 w_k1_MSFT_s016
                                 + 0.966811 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.07778 w_k1_ADM_s016
                                 + 1.39023 w_k1_AXP_s016
                                 + 1.82728 w_k1_DIS_s016
                                 + 1.34887 w_k1_FDX_s016
                                 + 1.56581 w_k1_IBM_s016
                                 + 1.4284 w_k1_INTC_s016 + 1.47391 w_k1_KO_s016
                                 + 1.27724 w_k1_MAT_s016
                                 + 1.20075 w_k1_MSFT_s016
                                 + 1.16961 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.772767 w_k1_ADM_s017
                                 + 0.939817 w_k1_AXP_s017
                                 + 0.918269 w_k1_DIS_s017
                                 + 0.843069 w_k1_FDX_s017
                                 + 1.13673 w_k1_IBM_s017
                                 + 0.916175 w_k1_INTC_s017
                                 + 0.939499 w_k1_KO_s017
                                 + 1.06832 w_k1_MAT_s017
                                 + 1.00933 w_k1_MSFT_s017
                                 + 0.85561 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.10754 w_k1_ADM_s017
                                 + 1.40547 w_k1_AXP_s017
                                 + 1.42537 w_k1_DIS_s017
                                 + 1.23094 w_k1_FDX_s017
                                 + 1.12553 w_k1_IBM_s017
                                 + 1.27262 w_k1_INTC_s017
                                 + 1.12097 w_k1_KO_s017 + 1.26251 w_k1_MAT_s017
                                 + 1.32654 w_k1_MSFT_s017
                                 + 1.04135 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.23329 w_k1_ADM_s017
                                 + 1.43087 w_k1_AXP_s017
                                 + 0.828244 w_k1_DIS_s017
                                 + 1.04425 w_k1_FDX_s017
                                 + 1.32386 w_k1_IBM_s017
                                 + 0.996267 w_k1_INTC_s017
                                 + 0.969946 w_k1_KO_s017
                                 + 1.39492 w_k1_MAT_s017
                                 + 0.946837 w_k1_MSFT_s017
                                 + 0.828075 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.88942 w_k1_ADM_s017
                                 + 0.693699 w_k1_AXP_s017
                                 + 1.11265 w_k1_DIS_s017
                                 + 1.00338 w_k1_FDX_s017
                                 + 0.684907 w_k1_IBM_s017
                                 + 1.07955 w_k1_INTC_s017
                                 + 0.936363 w_k1_KO_s017
                                 + 0.606878 w_k1_MAT_s017
                                 + 1.04798 w_k1_MSFT_s017
                                 + 0.882541 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.72663 w_k1_ADM_s017
                                 + 0.730874 w_k1_AXP_s017
                                 + 0.772128 w_k1_DIS_s017
                                 + 0.901316 w_k1_FDX_s017
                                 + 0.662326 w_k1_IBM_s017
                                 + 1.03451 w_k1_INTC_s017
                                 + 0.758219 w_k1_KO_s017
                                 + 0.606404 w_k1_MAT_s017
                                 + 0.979367 w_k1_MSFT_s017
                                 + 0.733578 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.01082 w_k1_ADM_s017
                                 + 1.04768 w_k1_AXP_s017
                                 + 1.06253 w_k1_DIS_s017 + 1.0449 w_k1_FDX_s017
                                 + 0.842132 w_k1_IBM_s017
                                 + 1.16098 w_k1_INTC_s017
                                 + 0.874148 w_k1_KO_s017
                                 + 0.729407 w_k1_MAT_s017
                                 + 1.24703 w_k1_MSFT_s017
                                 + 0.827926 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.874906 w_k1_ADM_s017
                                 + 0.972886 w_k1_AXP_s017
                                 + 0.825738 w_k1_DIS_s017
                                 + 0.807465 w_k1_FDX_s017
                                 + 1.46221 w_k1_IBM_s017
                                 + 1.03459 w_k1_INTC_s017
                                 + 0.989805 w_k1_KO_s017
                                 + 1.24367 w_k1_MAT_s017
                                 + 1.02581 w_k1_MSFT_s017
                                 + 0.83053 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.699654 w_k1_ADM_s017
                                 + 0.769629 w_k1_AXP_s017
                                 + 0.641209 w_k1_DIS_s017
                                 + 0.730697 w_k1_FDX_s017
                                 + 0.917746 w_k1_IBM_s017
                                 + 0.738499 w_k1_INTC_s017
                                 + 0.735407 w_k1_KO_s017
                                 + 0.899952 w_k1_MAT_s017
                                 + 0.821124 w_k1_MSFT_s017
                                 + 0.768989 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.09504 w_k1_ADM_s100
                                 + 0.943485 w_k1_AXP_s100
                                 + 1.10974 w_k1_DIS_s100
                                 + 1.01622 w_k1_FDX_s100
                                 + 1.60345 w_k1_IBM_s100
                                 + 0.933082 w_k1_INTC_s100
                                 + 1.28689 w_k1_KO_s100 + 1.7601 w_k1_MAT_s100
                                 + 0.92027 w_k1_MSFT_s100
                                 + 1.17338 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.01251 w_k1_ADM_s100
                                 + 1.15031 w_k1_AXP_s100
                                 + 1.01389 w_k1_DIS_s100 + 1.0574 w_k1_FDX_s100
                                 + 1.21553 w_k1_IBM_s100
                                 + 0.987107 w_k1_INTC_s100
                                 + 1.12535 w_k1_KO_s100 + 1.51518 w_k1_MAT_s100
                                 + 0.887605 w_k1_MSFT_s100
                                 + 1.0573 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.67455 w_k1_ADM_s100
                                 + 0.636674 w_k1_AXP_s100
                                 + 0.497598 w_k1_DIS_s100
                                 + 0.660523 w_k1_FDX_s100
                                 + 0.974906 w_k1_IBM_s100
                                 + 0.691334 w_k1_INTC_s100
                                 + 0.78305 w_k1_KO_s100 + 1.21191 w_k1_MAT_s100
                                 + 0.587688 w_k1_MSFT_s100
                                 + 0.829972 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.948761 w_k1_ADM_s100
                                 + 1.02842 w_k1_AXP_s100
                                 + 1.11808 w_k1_DIS_s100
                                 + 1.06365 w_k1_FDX_s100
                                 + 1.00808 w_k1_IBM_s100
                                 + 1.01937 w_k1_INTC_s100
                                 + 1.07884 w_k1_KO_s100 + 1.07556 w_k1_MAT_s100
                                 + 1.06326 w_k1_MSFT_s100
                                 + 1.02884 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.00316 w_k1_ADM_s100
                                 + 0.957153 w_k1_AXP_s100
                                 + 0.849161 w_k1_DIS_s100
                                 + 0.986961 w_k1_FDX_s100
                                 + 0.647864 w_k1_IBM_s100
                                 + 0.826976 w_k1_INTC_s100
                                 + 0.807943 w_k1_KO_s100
                                 + 0.902583 w_k1_MAT_s100
                                 + 0.934308 w_k1_MSFT_s100
                                 + 0.974152 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.12123 w_k1_ADM_s100
                                 + 1.11357 w_k1_AXP_s100
                                 + 1.74721 w_k1_DIS_s100
                                 + 1.48748 w_k1_FDX_s100 + 1.0829 w_k1_IBM_s100
                                 + 1.28105 w_k1_INTC_s100
                                 + 1.36765 w_k1_KO_s100 + 1.30301 w_k1_MAT_s100
                                 + 1.34829 w_k1_MSFT_s100
                                 + 1.28497 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.817047 w_k1_ADM_s100
                                 + 0.70483 w_k1_AXP_s100
                                 + 0.722026 w_k1_DIS_s100
                                 + 0.868015 w_k1_FDX_s100
                                 + 0.812898 w_k1_IBM_s100
                                 + 0.856859 w_k1_INTC_s100
                                 + 0.771679 w_k1_KO_s100
                                 + 0.902489 w_k1_MAT_s100
                                 + 0.842411 w_k1_MSFT_s100
                                 + 0.966052 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.836698 w_k1_ADM_s100
                                 + 0.85635 w_k1_AXP_s100
                                 + 0.70117 w_k1_DIS_s100
                                 + 0.762545 w_k1_FDX_s100
                                 + 1.08617 w_k1_IBM_s100
                                 + 0.71234 w_k1_INTC_s100
                                 + 1.03687 w_k1_KO_s100 + 1.1648 w_k1_MAT_s100
                                 + 0.66489 w_k1_MSFT_s100
                                 + 0.896075 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.02202 w_k1_ADM_s101
                                 + 1.02566 w_k1_AXP_s101
                                 + 0.924729 w_k1_DIS_s101
                                 + 1.0136 w_k1_FDX_s101
                                 + 0.686756 w_k1_IBM_s101
                                 + 0.82822 w_k1_INTC_s101
                                 + 0.864032 w_k1_KO_s101
                                 + 0.937267 w_k1_MAT_s101
                                 + 0.908692 w_k1_MSFT_s101
                                 + 0.937781 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.21242 w_k1_ADM_s101
                                 + 0.806719 w_k1_AXP_s101
                                 + 1.25417 w_k1_DIS_s101
                                 + 1.08463 w_k1_FDX_s101
                                 + 0.919378 w_k1_IBM_s101
                                 + 0.965065 w_k1_INTC_s101
                                 + 1.18742 w_k1_KO_s101
                                 + 0.977643 w_k1_MAT_s101
                                 + 1.26429 w_k1_MSFT_s101
                                 + 1.11911 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.914593 w_k1_ADM_s101
                                 + 0.723608 w_k1_AXP_s101
                                 + 0.931046 w_k1_DIS_s101
                                 + 0.874169 w_k1_FDX_s101
                                 + 0.97977 w_k1_IBM_s101
                                 + 0.864221 w_k1_INTC_s101
                                 + 1.05265 w_k1_KO_s101
                                 + 0.822339 w_k1_MAT_s101
                                 + 1.0249 w_k1_MSFT_s101
                                 + 0.987438 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.27678 w_k1_ADM_s101
                                 + 1.24197 w_k1_AXP_s101
                                 + 1.18062 w_k1_DIS_s101
                                 + 1.00202 w_k1_FDX_s101 + 1.4785 w_k1_IBM_s101
                                 + 0.939623 w_k1_INTC_s101
                                 + 1.23135 w_k1_KO_s101 + 1.37855 w_k1_MAT_s101
                                 + 1.03951 w_k1_MSFT_s101
                                 + 1.00271 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.07367 w_k1_ADM_s101
                                 + 0.823053 w_k1_AXP_s101
                                 + 0.663149 w_k1_DIS_s101
                                 + 0.795907 w_k1_FDX_s101
                                 + 0.732777 w_k1_IBM_s101
                                 + 0.729312 w_k1_INTC_s101
                                 + 0.742162 w_k1_KO_s101
                                 + 0.604341 w_k1_MAT_s101
                                 + 0.7759 w_k1_MSFT_s101
                                 + 0.749146 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.24813 w_k1_ADM_s101
                                 + 1.11464 w_k1_AXP_s101
                                 + 1.39119 w_k1_DIS_s101
                                 + 1.17843 w_k1_FDX_s101
                                 + 1.27829 w_k1_IBM_s101
                                 + 0.886569 w_k1_INTC_s101
                                 + 1.41873 w_k1_KO_s101 + 1.473 w_k1_MAT_s101
                                 + 1.15369 w_k1_MSFT_s101
                                 + 1.00892 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.4659 w_k1_ADM_s101
                                 + 1.1526 w_k1_AXP_s101 + 1.24838 w_k1_DIS_s101
                                 + 1.34995 w_k1_FDX_s101
                                 + 0.864448 w_k1_IBM_s101
                                 + 0.927631 w_k1_INTC_s101
                                 + 1.05789 w_k1_KO_s101
                                 + 0.847796 w_k1_MAT_s101
                                 + 1.10692 w_k1_MSFT_s101
                                 + 1.10374 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.37309 w_k1_ADM_s101
                                 + 1.34894 w_k1_AXP_s101
                                 + 1.42577 w_k1_DIS_s101
                                 + 1.27861 w_k1_FDX_s101
                                 + 1.13656 w_k1_IBM_s101
                                 + 0.822718 w_k1_INTC_s101
                                 + 1.16111 w_k1_KO_s101 + 1.496 w_k1_MAT_s101
                                 + 1.05468 w_k1_MSFT_s101
                                 + 1.05818 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.997717 w_k1_ADM_s102
                                 + 1.07422 w_k1_AXP_s102
                                 + 0.74619 w_k1_DIS_s102
                                 + 0.951352 w_k1_FDX_s102
                                 + 1.0473 w_k1_IBM_s102
                                 + 1.10665 w_k1_INTC_s102
                                 + 0.925792 w_k1_KO_s102
                                 + 1.57845 w_k1_MAT_s102
                                 + 0.949149 w_k1_MSFT_s102
                                 + 1.03638 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.11207 w_k1_ADM_s102
                                 + 1.24929 w_k1_AXP_s102
                                 + 0.837026 w_k1_DIS_s102
                                 + 0.976317 w_k1_FDX_s102
                                 + 1.25621 w_k1_IBM_s102
                                 + 0.919413 w_k1_INTC_s102
                                 + 0.887074 w_k1_KO_s102
                                 + 1.53882 w_k1_MAT_s102
                                 + 0.879234 w_k1_MSFT_s102
                                 + 1.02763 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.876661 w_k1_ADM_s102
                                 + 0.66267 w_k1_AXP_s102
                                 + 0.660005 w_k1_DIS_s102
                                 + 0.80173 w_k1_FDX_s102
                                 + 0.610578 w_k1_IBM_s102
                                 + 0.854427 w_k1_INTC_s102
                                 + 0.679314 w_k1_KO_s102
                                 + 0.598166 w_k1_MAT_s102
                                 + 0.861818 w_k1_MSFT_s102
                                 + 0.835511 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.2097 w_k1_ADM_s102
                                 + 1.22822 w_k1_AXP_s102
                                 + 1.08762 w_k1_DIS_s102
                                 + 1.16002 w_k1_FDX_s102
                                 + 0.910846 w_k1_IBM_s102
                                 + 0.958789 w_k1_INTC_s102
                                 + 0.937839 w_k1_KO_s102
                                 + 1.45745 w_k1_MAT_s102
                                 + 1.0269 w_k1_MSFT_s102
                                 + 0.961076 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.775435 w_k1_ADM_s102
                                 + 0.751255 w_k1_AXP_s102
                                 + 0.837576 w_k1_DIS_s102
                                 + 0.800292 w_k1_FDX_s102
                                 + 1.00207 w_k1_IBM_s102
                                 + 0.953495 w_k1_INTC_s102
                                 + 0.868107 w_k1_KO_s102
                                 + 0.849242 w_k1_MAT_s102
                                 + 0.938275 w_k1_MSFT_s102
                                 + 0.964009 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.17541 w_k1_ADM_s102
                                 + 1.62982 w_k1_AXP_s102
                                 + 1.35387 w_k1_DIS_s102 + 1.3915 w_k1_FDX_s102
                                 + 1.07559 w_k1_IBM_s102
                                 + 1.26201 w_k1_INTC_s102
                                 + 1.05394 w_k1_KO_s102 + 1.36517 w_k1_MAT_s102
                                 + 1.09465 w_k1_MSFT_s102
                                 + 1.23664 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.911624 w_k1_ADM_s102
                                 + 1.28634 w_k1_AXP_s102
                                 + 1.31914 w_k1_DIS_s102
                                 + 1.04479 w_k1_FDX_s102
                                 + 1.52304 w_k1_IBM_s102
                                 + 1.09788 w_k1_INTC_s102
                                 + 1.29622 w_k1_KO_s102 + 1.44886 w_k1_MAT_s102
                                 + 1.09251 w_k1_MSFT_s102
                                 + 1.15344 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.86005 w_k1_ADM_s102
                                 + 1.05759 w_k1_AXP_s102
                                 + 0.89017 w_k1_DIS_s102
                                 + 0.965765 w_k1_FDX_s102
                                 + 0.832983 w_k1_IBM_s102
                                 + 0.928526 w_k1_INTC_s102
                                 + 0.75565 w_k1_KO_s102 + 1.11917 w_k1_MAT_s102
                                 + 0.958026 w_k1_MSFT_s102
                                 + 0.906239 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.825966 w_k1_ADM_s103
                                 + 1.1379 w_k1_AXP_s103
                                 + 0.987142 w_k1_DIS_s103
                                 + 0.880418 w_k1_FDX_s103
                                 + 1.48231 w_k1_IBM_s103
                                 + 0.980953 w_k1_INTC_s103
                                 + 1.18069 w_k1_KO_s103 + 1.73381 w_k1_MAT_s103
                                 + 0.933953 w_k1_MSFT_s103
                                 + 1.04072 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.11752 w_k1_ADM_s103
                                 + 1.24496 w_k1_AXP_s103
                                 + 1.16316 w_k1_DIS_s103
                                 + 1.16086 w_k1_FDX_s103
                                 + 0.954361 w_k1_IBM_s103
                                 + 1.06776 w_k1_INTC_s103
                                 + 0.979927 w_k1_KO_s103
                                 + 0.778658 w_k1_MAT_s103
                                 + 1.19509 w_k1_MSFT_s103 + 1.0344 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.966711 w_k1_ADM_s103
                                 + 0.875783 w_k1_AXP_s103
                                 + 1.0695 w_k1_DIS_s103
                                 + 0.989108 w_k1_FDX_s103
                                 + 1.0539 w_k1_IBM_s103
                                 + 1.16168 w_k1_INTC_s103 + 1.183 w_k1_KO_s103
                                 + 0.922221 w_k1_MAT_s103
                                 + 1.20897 w_k1_MSFT_s103
                                 + 1.03399 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.775067 w_k1_ADM_s103
                                 + 0.815634 w_k1_AXP_s103
                                 + 0.743039 w_k1_DIS_s103
                                 + 0.838821 w_k1_FDX_s103
                                 + 0.784674 w_k1_IBM_s103
                                 + 0.904474 w_k1_INTC_s103
                                 + 0.914035 w_k1_KO_s103
                                 + 0.832343 w_k1_MAT_s103
                                 + 1.00197 w_k1_MSFT_s103
                                 + 0.874936 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.39116 w_k1_ADM_s103
                                 + 1.4703 w_k1_AXP_s103 + 1.44676 w_k1_DIS_s103
                                 + 1.2758 w_k1_FDX_s103 + 1.28513 w_k1_IBM_s103
                                 + 1.23448 w_k1_INTC_s103
                                 + 1.48003 w_k1_KO_s103 + 1.03981 w_k1_MAT_s103
                                 + 1.50999 w_k1_MSFT_s103
                                 + 1.25163 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.901913 w_k1_ADM_s103
                                 + 0.828235 w_k1_AXP_s103
                                 + 0.734166 w_k1_DIS_s103
                                 + 0.834891 w_k1_FDX_s103
                                 + 1.02942 w_k1_IBM_s103
                                 + 0.879139 w_k1_INTC_s103
                                 + 0.978195 w_k1_KO_s103
                                 + 0.722281 w_k1_MAT_s103
                                 + 0.89544 w_k1_MSFT_s103
                                 + 0.854754 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.06394 w_k1_ADM_s103
                                 + 1.24376 w_k1_AXP_s103
                                 + 1.04369 w_k1_DIS_s103
                                 + 1.06012 w_k1_FDX_s103
                                 + 1.47938 w_k1_IBM_s103
                                 + 0.986298 w_k1_INTC_s103
                                 + 1.18994 w_k1_KO_s103 + 1.26295 w_k1_MAT_s103
                                 + 1.2813 w_k1_MSFT_s103 + 1.1523 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.13912 w_k1_ADM_s103
                                 + 1.06258 w_k1_AXP_s103
                                 + 0.875306 w_k1_DIS_s103
                                 + 1.13198 w_k1_FDX_s103
                                 + 0.667513 w_k1_IBM_s103
                                 + 1.09237 w_k1_INTC_s103
                                 + 0.878494 w_k1_KO_s103
                                 + 0.726238 w_k1_MAT_s103
                                 + 1.12545 w_k1_MSFT_s103
                                 + 1.00629 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.10104 w_k1_ADM_s104
                                 + 1.12126 w_k1_AXP_s104
                                 + 1.29691 w_k1_DIS_s104
                                 + 1.30526 w_k1_FDX_s104
                                 + 0.837669 w_k1_IBM_s104
                                 + 1.28012 w_k1_INTC_s104
                                 + 1.07417 w_k1_KO_s104
                                 + 0.729298 w_k1_MAT_s104
                                 + 1.13041 w_k1_MSFT_s104
                                 + 1.15074 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.995134 w_k1_ADM_s104
                                 + 1.00301 w_k1_AXP_s104
                                 + 0.791139 w_k1_DIS_s104
                                 + 0.978801 w_k1_FDX_s104
                                 + 0.632064 w_k1_IBM_s104
                                 + 0.8789 w_k1_INTC_s104
                                 + 0.715038 w_k1_KO_s104
                                 + 0.690102 w_k1_MAT_s104
                                 + 0.910203 w_k1_MSFT_s104
                                 + 0.934906 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.888165 w_k1_ADM_s104
                                 + 0.657882 w_k1_AXP_s104
                                 + 0.609713 w_k1_DIS_s104
                                 + 0.717439 w_k1_FDX_s104
                                 + 0.987156 w_k1_IBM_s104
                                 + 0.75899 w_k1_INTC_s104 + 0.8059 w_k1_KO_s104
                                 + 0.798463 w_k1_MAT_s104
                                 + 0.738464 w_k1_MSFT_s104
                                 + 0.964985 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.24799 w_k1_ADM_s104
                                 + 1.02794 w_k1_AXP_s104
                                 + 0.806841 w_k1_DIS_s104
                                 + 1.06432 w_k1_FDX_s104
                                 + 0.862792 w_k1_IBM_s104
                                 + 0.866787 w_k1_INTC_s104
                                 + 0.797231 w_k1_KO_s104 + 1.03 w_k1_MAT_s104
                                 + 0.946793 w_k1_MSFT_s104
                                 + 1.04884 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.01585 w_k1_ADM_s104
                                 + 1.10359 w_k1_AXP_s104
                                 + 1.14278 w_k1_DIS_s104
                                 + 1.03027 w_k1_FDX_s104
                                 + 1.61725 w_k1_IBM_s104
                                 + 0.913513 w_k1_INTC_s104
                                 + 1.17605 w_k1_KO_s104 + 1.92951 w_k1_MAT_s104
                                 + 1.02136 w_k1_MSFT_s104
                                 + 1.01026 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.02239 w_k1_ADM_s104
                                 + 1.32695 w_k1_AXP_s104
                                 + 1.22118 w_k1_DIS_s104
                                 + 1.19556 w_k1_FDX_s104
                                 + 1.15281 w_k1_IBM_s104
                                 + 1.07301 w_k1_INTC_s104
                                 + 1.09835 w_k1_KO_s104 + 1.53862 w_k1_MAT_s104
                                 + 1.01483 w_k1_MSFT_s104
                                 + 1.09729 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.88546 w_k1_ADM_s104
                                 + 0.79711 w_k1_AXP_s104
                                 + 1.12147 w_k1_DIS_s104
                                 + 0.967473 w_k1_FDX_s104
                                 + 0.914803 w_k1_IBM_s104
                                 + 0.85479 w_k1_INTC_s104
                                 + 0.978904 w_k1_KO_s104
                                 + 0.843842 w_k1_MAT_s104
                                 + 0.954978 w_k1_MSFT_s104
                                 + 0.968162 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.939108 w_k1_ADM_s104
                                 + 0.693303 w_k1_AXP_s104
                                 + 0.68796 w_k1_DIS_s104
                                 + 0.843216 w_k1_FDX_s104
                                 + 0.733853 w_k1_IBM_s104
                                 + 0.83423 w_k1_INTC_s104
                                 + 0.758566 w_k1_KO_s104
                                 + 0.629906 w_k1_MAT_s104
                                 + 0.83562 w_k1_MSFT_s104
                                 + 0.996299 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.981514 w_k1_ADM_s105
                                 + 0.936998 w_k1_AXP_s105
                                 + 1.13973 w_k1_DIS_s105
                                 + 0.998437 w_k1_FDX_s105
                                 + 1.23839 w_k1_IBM_s105
                                 + 1.15024 w_k1_INTC_s105
                                 + 1.24558 w_k1_KO_s105
                                 + 0.845758 w_k1_MAT_s105
                                 + 1.14919 w_k1_MSFT_s105
                                 + 1.02728 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.847803 w_k1_ADM_s105
                                 + 0.763912 w_k1_AXP_s105
                                 + 0.576609 w_k1_DIS_s105
                                 + 0.73525 w_k1_FDX_s105
                                 + 0.870674 w_k1_IBM_s105
                                 + 0.680844 w_k1_INTC_s105
                                 + 0.732167 w_k1_KO_s105
                                 + 0.831667 w_k1_MAT_s105
                                 + 0.681167 w_k1_MSFT_s105
                                 + 0.851332 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.28085 w_k1_ADM_s105
                                 + 1.35719 w_k1_AXP_s105
                                 + 1.36668 w_k1_DIS_s105
                                 + 1.25603 w_k1_FDX_s105
                                 + 1.06249 w_k1_IBM_s105
                                 + 1.02327 w_k1_INTC_s105
                                 + 1.10477 w_k1_KO_s105 + 1.2864 w_k1_MAT_s105
                                 + 1.29105 w_k1_MSFT_s105
                                 + 0.956143 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.862672 w_k1_ADM_s105
                                 + 0.667357 w_k1_AXP_s105
                                 + 0.910341 w_k1_DIS_s105
                                 + 0.829102 w_k1_FDX_s105
                                 + 0.960598 w_k1_IBM_s105
                                 + 0.986659 w_k1_INTC_s105
                                 + 1.1063 w_k1_KO_s105 + 0.786305 w_k1_MAT_s105
                                 + 1.09657 w_k1_MSFT_s105
                                 + 0.936859 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.791292 w_k1_ADM_s105
                                 + 0.992417 w_k1_AXP_s105
                                 + 0.835475 w_k1_DIS_s105
                                 + 0.991846 w_k1_FDX_s105
                                 + 0.874134 w_k1_IBM_s105
                                 + 1.22185 w_k1_INTC_s105
                                 + 0.860305 w_k1_KO_s105
                                 + 0.726755 w_k1_MAT_s105
                                 + 1.10382 w_k1_MSFT_s105
                                 + 0.962527 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.17104 w_k1_ADM_s105
                                 + 1.34314 w_k1_AXP_s105
                                 + 1.48895 w_k1_DIS_s105
                                 + 1.15654 w_k1_FDX_s105 + 1.6218 w_k1_IBM_s105
                                 + 1.25282 w_k1_INTC_s105 + 1.3666 w_k1_KO_s105
                                 + 1.36158 w_k1_MAT_s105
                                 + 1.25375 w_k1_MSFT_s105
                                 + 1.21971 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.938829 w_k1_ADM_s105
                                 + 1.04926 w_k1_AXP_s105
                                 + 0.756035 w_k1_DIS_s105
                                 + 0.831456 w_k1_FDX_s105
                                 + 1.20454 w_k1_IBM_s105
                                 + 0.902689 w_k1_INTC_s105
                                 + 0.84246 w_k1_KO_s105 + 1.08699 w_k1_MAT_s105
                                 + 0.89388 w_k1_MSFT_s105
                                 + 0.905931 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.05125 w_k1_ADM_s105
                                 + 1.13394 w_k1_AXP_s105
                                 + 0.908966 w_k1_DIS_s105
                                 + 0.986776 w_k1_FDX_s105
                                 + 1.18911 w_k1_IBM_s105
                                 + 1.08459 w_k1_INTC_s105
                                 + 0.964404 w_k1_KO_s105
                                 + 0.877924 w_k1_MAT_s105
                                 + 1.02004 w_k1_MSFT_s105
                                 + 0.970771 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.40932 w_k1_ADM_s106
                                 + 1.14749 w_k1_AXP_s106
                                 + 1.07475 w_k1_DIS_s106
                                 + 1.20963 w_k1_FDX_s106
                                 + 0.732257 w_k1_IBM_s106
                                 + 1.08824 w_k1_INTC_s106
                                 + 0.998772 w_k1_KO_s106
                                 + 0.594657 w_k1_MAT_s106
                                 + 0.942778 w_k1_MSFT_s106
                                 + 0.960047 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.821451 w_k1_ADM_s106
                                 + 0.815297 w_k1_AXP_s106
                                 + 0.878127 w_k1_DIS_s106
                                 + 0.81897 w_k1_FDX_s106
                                 + 0.894776 w_k1_IBM_s106
                                 + 1.0002 w_k1_INTC_s106
                                 + 0.997632 w_k1_KO_s106
                                 + 0.833434 w_k1_MAT_s106
                                 + 0.841894 w_k1_MSFT_s106
                                 + 0.929629 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.866444 w_k1_ADM_s106
                                 + 0.651433 w_k1_AXP_s106
                                 + 0.592443 w_k1_DIS_s106
                                 + 0.698982 w_k1_FDX_s106
                                 + 0.676158 w_k1_IBM_s106
                                 + 0.900182 w_k1_INTC_s106
                                 + 0.680893 w_k1_KO_s106
                                 + 0.430797 w_k1_MAT_s106
                                 + 0.765405 w_k1_MSFT_s106
                                 + 0.767394 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.08415 w_k1_ADM_s106
                                 + 1.28465 w_k1_AXP_s106
                                 + 1.86086 w_k1_DIS_s106
                                 + 1.31414 w_k1_FDX_s106
                                 + 1.32822 w_k1_IBM_s106
                                 + 1.24747 w_k1_INTC_s106
                                 + 1.58945 w_k1_KO_s106 + 1.37631 w_k1_MAT_s106
                                 + 1.24556 w_k1_MSFT_s106
                                 + 1.35149 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.812269 w_k1_ADM_s106
                                 + 0.732273 w_k1_AXP_s106
                                 + 0.895845 w_k1_DIS_s106
                                 + 0.883236 w_k1_FDX_s106
                                 + 0.656317 w_k1_IBM_s106
                                 + 0.965443 w_k1_INTC_s106
                                 + 0.786121 w_k1_KO_s106
                                 + 0.521212 w_k1_MAT_s106
                                 + 0.982629 w_k1_MSFT_s106
                                 + 0.948389 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.961025 w_k1_ADM_s106
                                 + 1.13593 w_k1_AXP_s106
                                 + 0.907307 w_k1_DIS_s106
                                 + 0.849775 w_k1_FDX_s106
                                 + 1.43779 w_k1_IBM_s106
                                 + 1.07353 w_k1_INTC_s106
                                 + 1.16499 w_k1_KO_s106 + 1.28771 w_k1_MAT_s106
                                 + 0.841882 w_k1_MSFT_s106
                                 + 0.919056 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.19792 w_k1_ADM_s106
                                 + 0.988571 w_k1_AXP_s106
                                 + 1.10386 w_k1_DIS_s106
                                 + 1.00118 w_k1_FDX_s106
                                 + 1.07114 w_k1_IBM_s106
                                 + 1.20358 w_k1_INTC_s106
                                 + 1.08155 w_k1_KO_s106
                                 + 0.765744 w_k1_MAT_s106
                                 + 1.06492 w_k1_MSFT_s106
                                 + 1.01671 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.10105 w_k1_ADM_s106
                                 + 1.58488 w_k1_AXP_s106
                                 + 1.45205 w_k1_DIS_s106
                                 + 1.26022 w_k1_FDX_s106
                                 + 0.985093 w_k1_IBM_s106
                                 + 1.12577 w_k1_INTC_s106
                                 + 1.22182 w_k1_KO_s106 + 1.13252 w_k1_MAT_s106
                                 + 1.13731 w_k1_MSFT_s106
                                 + 1.13204 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.786207 w_k1_ADM_s107
                                 + 0.579073 w_k1_AXP_s107
                                 + 0.845331 w_k1_DIS_s107
                                 + 0.77089 w_k1_FDX_s107
                                 + 0.885671 w_k1_IBM_s107
                                 + 0.973551 w_k1_INTC_s107
                                 + 0.950845 w_k1_KO_s107
                                 + 0.901235 w_k1_MAT_s107
                                 + 0.932246 w_k1_MSFT_s107
                                 + 0.983245 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.58167 w_k1_ADM_s107
                                 + 1.14484 w_k1_AXP_s107
                                 + 1.33794 w_k1_DIS_s107
                                 + 1.13146 w_k1_FDX_s107
                                 + 1.49712 w_k1_IBM_s107
                                 + 1.17065 w_k1_INTC_s107
                                 + 1.51415 w_k1_KO_s107 + 1.54557 w_k1_MAT_s107
                                 + 1.44876 w_k1_MSFT_s107 + 1.3285 w_k1_PG_s107
                                 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.43728 w_k1_ADM_s107
                                 + 0.919136 w_k1_AXP_s107
                                 + 1.12439 w_k1_DIS_s107
                                 + 1.03209 w_k1_FDX_s107
                                 + 1.14701 w_k1_IBM_s107
                                 + 0.978325 w_k1_INTC_s107
                                 + 1.21201 w_k1_KO_s107 + 1.12258 w_k1_MAT_s107
                                 + 1.25622 w_k1_MSFT_s107
                                 + 1.15915 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.17754 w_k1_ADM_s107
                                 + 0.892553 w_k1_AXP_s107
                                 + 0.780905 w_k1_DIS_s107
                                 + 0.951901 w_k1_FDX_s107
                                 + 0.608365 w_k1_IBM_s107
                                 + 0.815184 w_k1_INTC_s107
                                 + 0.795299 w_k1_KO_s107
                                 + 0.92195 w_k1_MAT_s107
                                 + 0.985172 w_k1_MSFT_s107
                                 + 0.979954 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.55627 w_k1_ADM_s107
                                 + 1.91247 w_k1_AXP_s107
                                 + 2.18648 w_k1_DIS_s107
                                 + 1.46601 w_k1_FDX_s107 + 1.5187 w_k1_IBM_s107
                                 + 1.58613 w_k1_INTC_s107
                                 + 1.72774 w_k1_KO_s107 + 1.71481 w_k1_MAT_s107
                                 + 1.52193 w_k1_MSFT_s107
                                 + 1.36069 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.63697 w_k1_ADM_s107
                                 + 1.09132 w_k1_AXP_s107
                                 + 1.48203 w_k1_DIS_s107
                                 + 1.18536 w_k1_FDX_s107 + 1.1746 w_k1_IBM_s107
                                 + 1.16635 w_k1_INTC_s107 + 1.449 w_k1_KO_s107
                                 + 1.18865 w_k1_MAT_s107
                                 + 1.26823 w_k1_MSFT_s107
                                 + 1.22455 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.03865 w_k1_ADM_s107
                                 + 1.08213 w_k1_AXP_s107
                                 + 0.864378 w_k1_DIS_s107
                                 + 0.867153 w_k1_FDX_s107
                                 + 1.21641 w_k1_IBM_s107
                                 + 0.85336 w_k1_INTC_s107
                                 + 1.02264 w_k1_KO_s107 + 1.50606 w_k1_MAT_s107
                                 + 0.939314 w_k1_MSFT_s107
                                 + 1.03553 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.23427 w_k1_ADM_s107
                                 + 1.15358 w_k1_AXP_s107
                                 + 1.11825 w_k1_DIS_s107
                                 + 1.09238 w_k1_FDX_s107
                                 + 0.937848 w_k1_IBM_s107
                                 + 1.15269 w_k1_INTC_s107
                                 + 1.12364 w_k1_KO_s107 + 1.31999 w_k1_MAT_s107
                                 + 1.15438 w_k1_MSFT_s107
                                 + 1.14768 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.18257 w_k1_ADM_s110
                                 + 1.0347 w_k1_AXP_s110 + 1.13763 w_k1_DIS_s110
                                 + 1.22204 w_k1_FDX_s110
                                 + 0.770389 w_k1_IBM_s110
                                 + 1.29724 w_k1_INTC_s110
                                 + 0.962731 w_k1_KO_s110
                                 + 0.438234 w_k1_MAT_s110
                                 + 1.09308 w_k1_MSFT_s110 + 1.0363 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.677248 w_k1_ADM_s110
                                 + 0.66418 w_k1_AXP_s110
                                 + 0.552884 w_k1_DIS_s110
                                 + 0.72075 w_k1_FDX_s110
                                 + 0.752661 w_k1_IBM_s110
                                 + 0.692659 w_k1_INTC_s110
                                 + 0.683494 w_k1_KO_s110
                                 + 0.572284 w_k1_MAT_s110
                                 + 0.665338 w_k1_MSFT_s110
                                 + 0.823515 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.06011 w_k1_ADM_s110
                                 + 1.13653 w_k1_AXP_s110
                                 + 0.965338 w_k1_DIS_s110
                                 + 1.0148 w_k1_FDX_s110 + 1.49508 w_k1_IBM_s110
                                 + 0.963696 w_k1_INTC_s110
                                 + 1.13383 w_k1_KO_s110 + 1.27496 w_k1_MAT_s110
                                 + 0.928941 w_k1_MSFT_s110
                                 + 1.02647 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.00863 w_k1_ADM_s110
                                 + 0.884734 w_k1_AXP_s110
                                 + 1.05777 w_k1_DIS_s110
                                 + 0.95851 w_k1_FDX_s110
                                 + 1.11084 w_k1_IBM_s110
                                 + 0.95447 w_k1_INTC_s110 + 1.0891 w_k1_KO_s110
                                 + 0.934705 w_k1_MAT_s110
                                 + 0.993519 w_k1_MSFT_s110
                                 + 1.01583 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.972565 w_k1_ADM_s110
                                 + 0.87696 w_k1_AXP_s110
                                 + 0.827922 w_k1_DIS_s110
                                 + 0.899229 w_k1_FDX_s110
                                 + 0.773805 w_k1_IBM_s110
                                 + 0.887008 w_k1_INTC_s110
                                 + 0.761592 w_k1_KO_s110
                                 + 0.636286 w_k1_MAT_s110
                                 + 0.928675 w_k1_MSFT_s110
                                 + 0.868001 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18727 w_k1_ADM_s110
                                 + 1.19896 w_k1_AXP_s110 + 1.3107 w_k1_DIS_s110
                                 + 1.26548 w_k1_FDX_s110
                                 + 1.01963 w_k1_IBM_s110
                                 + 1.20989 w_k1_INTC_s110
                                 + 1.11569 w_k1_KO_s110
                                 + 0.958319 w_k1_MAT_s110
                                 + 1.16172 w_k1_MSFT_s110
                                 + 1.14879 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.39574 w_k1_ADM_s110
                                 + 1.44169 w_k1_AXP_s110
                                 + 1.42606 w_k1_DIS_s110
                                 + 1.35658 w_k1_FDX_s110
                                 + 1.39581 w_k1_IBM_s110
                                 + 1.13323 w_k1_INTC_s110
                                 + 1.22914 w_k1_KO_s110 + 1.29523 w_k1_MAT_s110
                                 + 1.04465 w_k1_MSFT_s110
                                 + 1.22448 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.859547 w_k1_ADM_s110
                                 + 0.872523 w_k1_AXP_s110
                                 + 0.775185 w_k1_DIS_s110
                                 + 0.835698 w_k1_FDX_s110
                                 + 1.01726 w_k1_IBM_s110
                                 + 0.782486 w_k1_INTC_s110
                                 + 0.871237 w_k1_KO_s110
                                 + 0.911516 w_k1_MAT_s110
                                 + 0.831182 w_k1_MSFT_s110
                                 + 0.925388 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.849997 w_k1_ADM_s111
                                 + 0.838156 w_k1_AXP_s111
                                 + 0.784322 w_k1_DIS_s111
                                 + 0.905283 w_k1_FDX_s111
                                 + 0.883952 w_k1_IBM_s111
                                 + 0.888287 w_k1_INTC_s111
                                 + 0.918297 w_k1_KO_s111
                                 + 0.820817 w_k1_MAT_s111
                                 + 0.826076 w_k1_MSFT_s111
                                 + 0.954253 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.18373 w_k1_ADM_s111
                                 + 1.36054 w_k1_AXP_s111
                                 + 1.76104 w_k1_DIS_s111 + 1.4953 w_k1_FDX_s111
                                 + 1.19209 w_k1_IBM_s111
                                 + 1.10591 w_k1_INTC_s111
                                 + 1.41533 w_k1_KO_s111 + 1.3536 w_k1_MAT_s111
                                 + 1.25264 w_k1_MSFT_s111
                                 + 1.13822 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.02996 w_k1_ADM_s111
                                 + 1.12916 w_k1_AXP_s111
                                 + 1.35323 w_k1_DIS_s111
                                 + 1.14575 w_k1_FDX_s111
                                 + 1.09532 w_k1_IBM_s111
                                 + 1.07371 w_k1_INTC_s111
                                 + 1.30296 w_k1_KO_s111 + 1.07101 w_k1_MAT_s111
                                 + 1.05601 w_k1_MSFT_s111
                                 + 1.14905 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.02112 w_k1_ADM_s111
                                 + 0.959642 w_k1_AXP_s111
                                 + 0.874989 w_k1_DIS_s111
                                 + 0.935499 w_k1_FDX_s111
                                 + 1.03531 w_k1_IBM_s111
                                 + 0.897606 w_k1_INTC_s111
                                 + 0.953789 w_k1_KO_s111
                                 + 1.03687 w_k1_MAT_s111
                                 + 0.880928 w_k1_MSFT_s111
                                 + 0.905028 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.00685 w_k1_ADM_s111
                                 + 1.08833 w_k1_AXP_s111
                                 + 1.30642 w_k1_DIS_s111
                                 + 1.37671 w_k1_FDX_s111
                                 + 0.65607 w_k1_IBM_s111
                                 + 0.989492 w_k1_INTC_s111
                                 + 0.96716 w_k1_KO_s111
                                 + 0.646211 w_k1_MAT_s111
                                 + 1.13829 w_k1_MSFT_s111
                                 + 0.890274 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.8284 w_k1_ADM_s111
                                 + 0.704794 w_k1_AXP_s111
                                 + 0.70564 w_k1_DIS_s111
                                 + 0.790406 w_k1_FDX_s111
                                 + 0.675252 w_k1_IBM_s111
                                 + 0.879646 w_k1_INTC_s111
                                 + 0.775592 w_k1_KO_s111
                                 + 0.498123 w_k1_MAT_s111
                                 + 0.768861 w_k1_MSFT_s111
                                 + 0.840508 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.04909 w_k1_ADM_s111
                                 + 0.923158 w_k1_AXP_s111
                                 + 1.03601 w_k1_DIS_s111 + 1.1149 w_k1_FDX_s111
                                 + 1.04806 w_k1_IBM_s111
                                 + 1.11696 w_k1_INTC_s111
                                 + 1.02557 w_k1_KO_s111
                                 + 0.655502 w_k1_MAT_s111
                                 + 0.992278 w_k1_MSFT_s111
                                 + 0.98919 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.08264 w_k1_ADM_s111
                                 + 1.38103 w_k1_AXP_s111
                                 + 1.25133 w_k1_DIS_s111
                                 + 1.01916 w_k1_FDX_s111
                                 + 1.64032 w_k1_IBM_s111
                                 + 0.907988 w_k1_INTC_s111
                                 + 1.50865 w_k1_KO_s111 + 2.27561 w_k1_MAT_s111
                                 + 0.90257 w_k1_MSFT_s111
                                 + 1.18686 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07757 w_k1_ADM_s112
                                 + 0.818865 w_k1_AXP_s112
                                 + 0.643609 w_k1_DIS_s112
                                 + 0.842403 w_k1_FDX_s112
                                 + 0.692576 w_k1_IBM_s112
                                 + 0.676836 w_k1_INTC_s112
                                 + 0.660531 w_k1_KO_s112
                                 + 0.610963 w_k1_MAT_s112
                                 + 0.827536 w_k1_MSFT_s112
                                 + 0.854889 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.15356 w_k1_ADM_s112
                                 + 0.816717 w_k1_AXP_s112
                                 + 1.35324 w_k1_DIS_s112
                                 + 1.15851 w_k1_FDX_s112
                                 + 0.859446 w_k1_IBM_s112
                                 + 1.35295 w_k1_INTC_s112
                                 + 1.02691 w_k1_KO_s112
                                 + 0.586285 w_k1_MAT_s112
                                 + 1.34756 w_k1_MSFT_s112
                                 + 0.983376 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.05834 w_k1_ADM_s112
                                 + 1.33318 w_k1_AXP_s112 + 1.2799 w_k1_DIS_s112
                                 + 1.27013 w_k1_FDX_s112
                                 + 1.14964 w_k1_IBM_s112
                                 + 0.99595 w_k1_INTC_s112
                                 + 1.07631 w_k1_KO_s112 + 1.35379 w_k1_MAT_s112
                                 + 1.21633 w_k1_MSFT_s112
                                 + 1.12959 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.3757 w_k1_ADM_s112
                                 + 1.36587 w_k1_AXP_s112
                                 + 1.61085 w_k1_DIS_s112
                                 + 1.41927 w_k1_FDX_s112
                                 + 1.16587 w_k1_IBM_s112
                                 + 1.32685 w_k1_INTC_s112
                                 + 1.22666 w_k1_KO_s112 + 1.15488 w_k1_MAT_s112
                                 + 1.45442 w_k1_MSFT_s112
                                 + 1.13476 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.332 w_k1_ADM_s112
                                 + 0.939511 w_k1_AXP_s112
                                 + 0.94245 w_k1_DIS_s112
                                 + 1.12029 w_k1_FDX_s112
                                 + 0.720957 w_k1_IBM_s112
                                 + 1.1686 w_k1_INTC_s112 + 0.94479 w_k1_KO_s112
                                 + 0.722709 w_k1_MAT_s112
                                 + 1.21373 w_k1_MSFT_s112
                                 + 1.02181 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.07793 w_k1_ADM_s112
                                 + 0.738626 w_k1_AXP_s112
                                 + 0.852269 w_k1_DIS_s112
                                 + 0.801555 w_k1_FDX_s112
                                 + 0.992201 w_k1_IBM_s112
                                 + 0.784256 w_k1_INTC_s112
                                 + 0.97839 w_k1_KO_s112
                                 + 0.633689 w_k1_MAT_s112
                                 + 0.925275 w_k1_MSFT_s112
                                 + 0.92539 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.04462 w_k1_ADM_s112
                                 + 0.744146 w_k1_AXP_s112
                                 + 0.982577 w_k1_DIS_s112
                                 + 0.946151 w_k1_FDX_s112
                                 + 0.761257 w_k1_IBM_s112
                                 + 0.835946 w_k1_INTC_s112
                                 + 0.824862 w_k1_KO_s112
                                 + 0.637952 w_k1_MAT_s112
                                 + 0.976224 w_k1_MSFT_s112
                                 + 0.931699 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.02667 w_k1_ADM_s112
                                 + 0.900442 w_k1_AXP_s112
                                 + 0.884203 w_k1_DIS_s112
                                 + 0.944399 w_k1_FDX_s112
                                 + 1.17823 w_k1_IBM_s112
                                 + 0.717547 w_k1_INTC_s112
                                 + 0.948428 w_k1_KO_s112
                                 + 1.11408 w_k1_MAT_s112
                                 + 0.840504 w_k1_MSFT_s112
                                 + 0.952738 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.34966 w_k1_ADM_s113
                                 + 1.21478 w_k1_AXP_s113
                                 + 1.07742 w_k1_DIS_s113
                                 + 1.19869 w_k1_FDX_s113
                                 + 0.835447 w_k1_IBM_s113
                                 + 1.09824 w_k1_INTC_s113
                                 + 0.908788 w_k1_KO_s113
                                 + 1.00948 w_k1_MAT_s113
                                 + 1.01403 w_k1_MSFT_s113
                                 + 1.05142 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.38287 w_k1_ADM_s113
                                 + 1.30323 w_k1_AXP_s113
                                 + 1.37453 w_k1_DIS_s113
                                 + 1.16703 w_k1_FDX_s113
                                 + 1.73587 w_k1_IBM_s113
                                 + 1.32477 w_k1_INTC_s113
                                 + 1.37165 w_k1_KO_s113 + 1.80176 w_k1_MAT_s113
                                 + 1.27812 w_k1_MSFT_s113
                                 + 1.29527 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.997901 w_k1_ADM_s113
                                 + 1.09909 w_k1_AXP_s113
                                 + 1.10515 w_k1_DIS_s113
                                 + 1.08008 w_k1_FDX_s113
                                 + 1.08767 w_k1_IBM_s113
                                 + 1.21694 w_k1_INTC_s113
                                 + 1.07126 w_k1_KO_s113 + 1.18637 w_k1_MAT_s113
                                 + 0.952461 w_k1_MSFT_s113
                                 + 1.19355 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.851862 w_k1_ADM_s113
                                 + 0.939895 w_k1_AXP_s113
                                 + 0.753831 w_k1_DIS_s113
                                 + 0.929283 w_k1_FDX_s113
                                 + 0.571031 w_k1_IBM_s113
                                 + 0.966161 w_k1_INTC_s113
                                 + 0.564539 w_k1_KO_s113
                                 + 0.524921 w_k1_MAT_s113
                                 + 0.94606 w_k1_MSFT_s113
                                 + 0.786847 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.33689 w_k1_ADM_s113
                                 + 1.69999 w_k1_AXP_s113
                                 + 1.61509 w_k1_DIS_s113
                                 + 1.54123 w_k1_FDX_s113
                                 + 1.01194 w_k1_IBM_s113
                                 + 1.65221 w_k1_INTC_s113
                                 + 1.24999 w_k1_KO_s113 + 1.35993 w_k1_MAT_s113
                                 + 1.34426 w_k1_MSFT_s113
                                 + 1.33703 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.673942 w_k1_ADM_s113
                                 + 0.770219 w_k1_AXP_s113
                                 + 0.797869 w_k1_DIS_s113
                                 + 0.814177 w_k1_FDX_s113
                                 + 1.08903 w_k1_IBM_s113
                                 + 0.975498 w_k1_INTC_s113
                                 + 0.852044 w_k1_KO_s113
                                 + 1.09225 w_k1_MAT_s113
                                 + 0.970944 w_k1_MSFT_s113
                                 + 0.955999 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.89779 w_k1_ADM_s113
                                 + 0.822973 w_k1_AXP_s113
                                 + 0.880036 w_k1_DIS_s113
                                 + 0.957579 w_k1_FDX_s113
                                 + 0.742622 w_k1_IBM_s113
                                 + 0.909011 w_k1_INTC_s113
                                 + 0.788069 w_k1_KO_s113
                                 + 0.785952 w_k1_MAT_s113
                                 + 0.925464 w_k1_MSFT_s113
                                 + 0.903056 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.622228 w_k1_ADM_s113
                                 + 0.595961 w_k1_AXP_s113
                                 + 0.554405 w_k1_DIS_s113
                                 + 0.647678 w_k1_FDX_s113
                                 + 0.773521 w_k1_IBM_s113
                                 + 0.744962 w_k1_INTC_s113
                                 + 0.657133 w_k1_KO_s113
                                 + 0.764133 w_k1_MAT_s113
                                 + 0.767827 w_k1_MSFT_s113
                                 + 0.699393 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.5212 w_k1_ADM_s114
                                 + 1.51798 w_k1_AXP_s114
                                 + 1.53648 w_k1_DIS_s114
                                 + 1.31896 w_k1_FDX_s114
                                 + 1.47535 w_k1_IBM_s114
                                 + 0.990821 w_k1_INTC_s114
                                 + 1.22799 w_k1_KO_s114 + 1.64978 w_k1_MAT_s114
                                 + 1.07136 w_k1_MSFT_s114
                                 + 1.36206 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.38779 w_k1_ADM_s114
                                 + 1.0652 w_k1_AXP_s114 + 1.31162 w_k1_DIS_s114
                                 + 1.05329 w_k1_FDX_s114
                                 + 1.75299 w_k1_IBM_s114
                                 + 0.918582 w_k1_INTC_s114
                                 + 1.35067 w_k1_KO_s114 + 1.56451 w_k1_MAT_s114
                                 + 1.19259 w_k1_MSFT_s114 + 1.0697 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.31198 w_k1_ADM_s114
                                 + 1.05482 w_k1_AXP_s114 + 1.0838 w_k1_DIS_s114
                                 + 1.10876 w_k1_FDX_s114
                                 + 0.881017 w_k1_IBM_s114
                                 + 1.02843 w_k1_INTC_s114
                                 + 0.863375 w_k1_KO_s114
                                 + 0.750412 w_k1_MAT_s114
                                 + 1.03005 w_k1_MSFT_s114
                                 + 1.05197 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.976993 w_k1_ADM_s114
                                 + 0.821529 w_k1_AXP_s114
                                 + 0.75859 w_k1_DIS_s114
                                 + 0.721967 w_k1_FDX_s114
                                 + 1.32941 w_k1_IBM_s114
                                 + 0.629276 w_k1_INTC_s114
                                 + 0.74682 w_k1_KO_s114 + 1.32769 w_k1_MAT_s114
                                 + 0.672863 w_k1_MSFT_s114
                                 + 0.895355 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19547 w_k1_ADM_s114
                                 + 0.999093 w_k1_AXP_s114
                                 + 1.06957 w_k1_DIS_s114
                                 + 0.954285 w_k1_FDX_s114
                                 + 1.12822 w_k1_IBM_s114
                                 + 0.831705 w_k1_INTC_s114
                                 + 0.881741 w_k1_KO_s114
                                 + 1.26367 w_k1_MAT_s114
                                 + 0.942737 w_k1_MSFT_s114
                                 + 0.959571 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.965663 w_k1_ADM_s114
                                 + 0.781821 w_k1_AXP_s114
                                 + 0.803435 w_k1_DIS_s114
                                 + 0.862125 w_k1_FDX_s114
                                 + 0.593998 w_k1_IBM_s114
                                 + 0.838451 w_k1_INTC_s114
                                 + 0.653026 w_k1_KO_s114
                                 + 0.495143 w_k1_MAT_s114
                                 + 0.833441 w_k1_MSFT_s114
                                 + 0.877898 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.3732 w_k1_ADM_s114
                                 + 1.24049 w_k1_AXP_s114 + 1.3967 w_k1_DIS_s114
                                 + 1.28439 w_k1_FDX_s114
                                 + 1.14955 w_k1_IBM_s114
                                 + 1.05855 w_k1_INTC_s114 + 1.0785 w_k1_KO_s114
                                 + 1.30009 w_k1_MAT_s114
                                 + 1.21529 w_k1_MSFT_s114
                                 + 1.23501 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.774334 w_k1_ADM_s114
                                 + 0.700566 w_k1_AXP_s114
                                 + 0.764456 w_k1_DIS_s114
                                 + 0.764984 w_k1_FDX_s114
                                 + 0.818609 w_k1_IBM_s114
                                 + 0.700283 w_k1_INTC_s114
                                 + 0.672587 w_k1_KO_s114
                                 + 0.684234 w_k1_MAT_s114
                                 + 0.687076 w_k1_MSFT_s114
                                 + 0.87659 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.19112 w_k1_ADM_s115
                                 + 1.00397 w_k1_AXP_s115
                                 + 1.26978 w_k1_DIS_s115
                                 + 1.34516 w_k1_FDX_s115
                                 + 0.746698 w_k1_IBM_s115
                                 + 1.20351 w_k1_INTC_s115 + 1.2051 w_k1_KO_s115
                                 + 1.0456 w_k1_MAT_s115
                                 + 1.38362 w_k1_MSFT_s115
                                 + 1.21182 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.17217 w_k1_ADM_s115
                                 + 1.18796 w_k1_AXP_s115
                                 + 1.05149 w_k1_DIS_s115
                                 + 1.28118 w_k1_FDX_s115
                                 + 0.993458 w_k1_IBM_s115
                                 + 0.979783 w_k1_INTC_s115
                                 + 1.0729 w_k1_KO_s115 + 0.837454 w_k1_MAT_s115
                                 + 1.15022 w_k1_MSFT_s115
                                 + 1.18765 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.1406 w_k1_ADM_s115
                                 + 1.02379 w_k1_AXP_s115
                                 + 0.878688 w_k1_DIS_s115
                                 + 1.06905 w_k1_FDX_s115
                                 + 0.879323 w_k1_IBM_s115
                                 + 0.79792 w_k1_INTC_s115
                                 + 0.928634 w_k1_KO_s115
                                 + 0.890893 w_k1_MAT_s115
                                 + 0.975348 w_k1_MSFT_s115
                                 + 0.885572 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.876823 w_k1_ADM_s115
                                 + 0.72409 w_k1_AXP_s115
                                 + 0.726997 w_k1_DIS_s115
                                 + 0.818578 w_k1_FDX_s115
                                 + 1.10845 w_k1_IBM_s115
                                 + 0.85649 w_k1_INTC_s115
                                 + 0.891988 w_k1_KO_s115
                                 + 1.18142 w_k1_MAT_s115
                                 + 0.88167 w_k1_MSFT_s115
                                 + 1.04401 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.07124 w_k1_ADM_s115
                                 + 0.881399 w_k1_AXP_s115
                                 + 0.968538 w_k1_DIS_s115
                                 + 0.926124 w_k1_FDX_s115
                                 + 1.36898 w_k1_IBM_s115
                                 + 0.793549 w_k1_INTC_s115
                                 + 1.2384 w_k1_KO_s115 + 1.23751 w_k1_MAT_s115
                                 + 1.00053 w_k1_MSFT_s115 + 1.1338 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.893925 w_k1_ADM_s115
                                 + 0.750904 w_k1_AXP_s115
                                 + 0.650712 w_k1_DIS_s115
                                 + 0.855532 w_k1_FDX_s115
                                 + 0.727883 w_k1_IBM_s115
                                 + 0.789134 w_k1_INTC_s115
                                 + 0.766753 w_k1_KO_s115
                                 + 0.736519 w_k1_MAT_s115
                                 + 0.824603 w_k1_MSFT_s115
                                 + 0.960271 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.1098 w_k1_ADM_s115
                                 + 0.851792 w_k1_AXP_s115
                                 + 1.23631 w_k1_DIS_s115 + 1.0381 w_k1_FDX_s115
                                 + 1.07873 w_k1_IBM_s115
                                 + 1.00895 w_k1_INTC_s115
                                 + 1.03019 w_k1_KO_s115
                                 + 0.998898 w_k1_MAT_s115
                                 + 1.15828 w_k1_MSFT_s115 + 1.1034 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.17037 w_k1_ADM_s115
                                 + 1.29022 w_k1_AXP_s115
                                 + 1.34292 w_k1_DIS_s115
                                 + 1.09904 w_k1_FDX_s115
                                 + 1.58171 w_k1_IBM_s115
                                 + 0.901802 w_k1_INTC_s115
                                 + 1.52188 w_k1_KO_s115 + 2.0567 w_k1_MAT_s115
                                 + 1.18567 w_k1_MSFT_s115
                                 + 1.26854 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.927815 w_k1_ADM_s116
                                 + 0.929567 w_k1_AXP_s116
                                 + 0.897625 w_k1_DIS_s116
                                 + 0.88416 w_k1_FDX_s116
                                 + 1.02777 w_k1_IBM_s116
                                 + 0.997649 w_k1_INTC_s116
                                 + 1.05601 w_k1_KO_s116
                                 + 0.923215 w_k1_MAT_s116
                                 + 0.964728 w_k1_MSFT_s116
                                 + 0.897453 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.10002 w_k1_ADM_s116
                                 + 0.774498 w_k1_AXP_s116
                                 + 1.14354 w_k1_DIS_s116
                                 + 1.13064 w_k1_FDX_s116
                                 + 0.783673 w_k1_IBM_s116
                                 + 1.11701 w_k1_INTC_s116
                                 + 0.970102 w_k1_KO_s116
                                 + 0.528533 w_k1_MAT_s116
                                 + 1.16143 w_k1_MSFT_s116
                                 + 0.974666 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.831144 w_k1_ADM_s116
                                 + 0.831224 w_k1_AXP_s116
                                 + 0.700077 w_k1_DIS_s116
                                 + 0.748077 w_k1_FDX_s116
                                 + 1.1888 w_k1_IBM_s116
                                 + 0.795009 w_k1_INTC_s116
                                 + 0.897284 w_k1_KO_s116
                                 + 0.917354 w_k1_MAT_s116
                                 + 0.762358 w_k1_MSFT_s116
                                 + 0.887201 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.46181 w_k1_ADM_s116
                                 + 1.42365 w_k1_AXP_s116
                                 + 1.42851 w_k1_DIS_s116
                                 + 1.36431 w_k1_FDX_s116
                                 + 1.03835 w_k1_IBM_s116
                                 + 1.23618 w_k1_INTC_s116
                                 + 1.27098 w_k1_KO_s116 + 1.05597 w_k1_MAT_s116
                                 + 1.28626 w_k1_MSFT_s116
                                 + 1.20656 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.931474 w_k1_ADM_s116
                                 + 0.835423 w_k1_AXP_s116
                                 + 0.845463 w_k1_DIS_s116
                                 + 0.965527 w_k1_FDX_s116
                                 + 0.583649 w_k1_IBM_s116
                                 + 1.03199 w_k1_INTC_s116
                                 + 0.904811 w_k1_KO_s116
                                 + 0.645448 w_k1_MAT_s116
                                 + 0.996802 w_k1_MSFT_s116
                                 + 0.92418 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.841045 w_k1_ADM_s116
                                 + 0.709536 w_k1_AXP_s116
                                 + 0.981856 w_k1_DIS_s116
                                 + 0.803323 w_k1_FDX_s116
                                 + 0.848322 w_k1_IBM_s116
                                 + 0.809917 w_k1_INTC_s116
                                 + 0.920808 w_k1_KO_s116
                                 + 0.667507 w_k1_MAT_s116
                                 + 0.892277 w_k1_MSFT_s116
                                 + 0.836447 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.17816 w_k1_ADM_s116
                                 + 1.0674 w_k1_AXP_s116 + 1.10994 w_k1_DIS_s116
                                 + 1.07706 w_k1_FDX_s116 + 1.0657 w_k1_IBM_s116
                                 + 0.962059 w_k1_INTC_s116
                                 + 0.951513 w_k1_KO_s116
                                 + 1.01006 w_k1_MAT_s116
                                 + 1.05696 w_k1_MSFT_s116
                                 + 1.01934 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.27048 w_k1_ADM_s116
                                 + 1.45001 w_k1_AXP_s116
                                 + 1.36611 w_k1_DIS_s116
                                 + 1.07069 w_k1_FDX_s116
                                 + 1.47575 w_k1_IBM_s116
                                 + 1.22736 w_k1_INTC_s116
                                 + 1.38719 w_k1_KO_s116 + 1.32501 w_k1_MAT_s116
                                 + 1.07729 w_k1_MSFT_s116
                                 + 1.09209 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.687032 w_k1_ADM_s117
                                 + 0.688379 w_k1_AXP_s117
                                 + 0.595638 w_k1_DIS_s117
                                 + 0.745496 w_k1_FDX_s117
                                 + 0.742568 w_k1_IBM_s117
                                 + 0.878628 w_k1_INTC_s117
                                 + 0.730414 w_k1_KO_s117
                                 + 0.552862 w_k1_MAT_s117
                                 + 0.749607 w_k1_MSFT_s117
                                 + 0.824429 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.70116 w_k1_ADM_s117
                                 + 0.772242 w_k1_AXP_s117
                                 + 0.799942 w_k1_DIS_s117
                                 + 0.739646 w_k1_FDX_s117
                                 + 1.02978 w_k1_IBM_s117
                                 + 0.848425 w_k1_INTC_s117
                                 + 0.860518 w_k1_KO_s117
                                 + 0.775812 w_k1_MAT_s117
                                 + 0.724495 w_k1_MSFT_s117
                                 + 0.932267 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.11431 w_k1_ADM_s117
                                 + 1.20261 w_k1_AXP_s117
                                 + 1.37495 w_k1_DIS_s117
                                 + 1.04016 w_k1_FDX_s117
                                 + 1.60936 w_k1_IBM_s117 + 1.144 w_k1_INTC_s117
                                 + 1.48412 w_k1_KO_s117 + 1.51018 w_k1_MAT_s117
                                 + 1.05238 w_k1_MSFT_s117
                                 + 1.11144 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.677053 w_k1_ADM_s117
                                 + 0.749906 w_k1_AXP_s117
                                 + 0.892475 w_k1_DIS_s117
                                 + 0.917111 w_k1_FDX_s117
                                 + 0.532188 w_k1_IBM_s117
                                 + 0.955618 w_k1_INTC_s117
                                 + 0.655508 w_k1_KO_s117 + 0.3546 w_k1_MAT_s117
                                 + 0.837283 w_k1_MSFT_s117
                                 + 0.915774 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.844849 w_k1_ADM_s117
                                 + 1.03347 w_k1_AXP_s117
                                 + 0.846819 w_k1_DIS_s117
                                 + 0.972599 w_k1_FDX_s117
                                 + 0.751735 w_k1_IBM_s117
                                 + 1.08931 w_k1_INTC_s117
                                 + 0.871879 w_k1_KO_s117
                                 + 0.655512 w_k1_MAT_s117
                                 + 0.888636 w_k1_MSFT_s117
                                 + 0.950421 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.81615 w_k1_ADM_s117
                                 + 1.1256 w_k1_AXP_s117
                                 + 0.932968 w_k1_DIS_s117
                                 + 0.948123 w_k1_FDX_s117
                                 + 1.08045 w_k1_IBM_s117
                                 + 0.987146 w_k1_INTC_s117
                                 + 1.06539 w_k1_KO_s117 + 0.98373 w_k1_MAT_s117
                                 + 0.859619 w_k1_MSFT_s117
                                 + 1.07124 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.696234 w_k1_ADM_s117
                                 + 0.865659 w_k1_AXP_s117
                                 + 0.761549 w_k1_DIS_s117
                                 + 0.743374 w_k1_FDX_s117
                                 + 1.42323 w_k1_IBM_s117
                                 + 1.0045 w_k1_INTC_s117 + 1.0386 w_k1_KO_s117
                                 + 1.21342 w_k1_MAT_s117
                                 + 0.835369 w_k1_MSFT_s117
                                 + 1.03616 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.04196 w_k1_ADM_s117
                                 + 1.19263 w_k1_AXP_s117
                                 + 1.46662 w_k1_DIS_s117
                                 + 1.16811 w_k1_FDX_s117
                                 + 1.01699 w_k1_IBM_s117
                                 + 1.19852 w_k1_INTC_s117
                                 + 1.31951 w_k1_KO_s117 + 1.00796 w_k1_MAT_s117
                                 + 1.15791 w_k1_MSFT_s117
                                 + 1.21204 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0268396978071059 w_k1_ADM_s000
                                 + 0.0265117545915848 w_k1_ADM_s001
                                 + 0.0320021036200668 w_k1_ADM_s002
                                 + 0.0404652960293209 w_k1_ADM_s003
                                 + 0.0220651963805481 w_k1_ADM_s004
                                 + 0.029082456019825 w_k1_ADM_s005
                                 + 0.0542367774124319 w_k1_ADM_s006
                                 + 0.0122677402463853 w_k1_ADM_s007
                                 + 0.0260670805718214 w_k1_ADM_s010
                                 + 0.0261070885316404 w_k1_ADM_s011
                                 + 0.0124450209458814 w_k1_ADM_s012
                                 + 0.0171777291903485 w_k1_ADM_s013
                                 + 0.0076082959708258 w_k1_ADM_s014
                                 + 0.0363081959506047 w_k1_ADM_s015
                                 + 0.015415510478911 w_k1_ADM_s016
                                 + 0.0120716955689675 w_k1_ADM_s017
                                 + 0.0219331769726908 w_k1_ADM_s100
                                 + 0.0532602002733812 w_k1_ADM_s101
                                 + 0.0150472539214304 w_k1_ADM_s102
                                 + 0.0380397554887239 w_k1_ADM_s103
                                 + 0.031503445959989 w_k1_ADM_s104
                                 + 0.0284126774208952 w_k1_ADM_s105
                                 + 0.0531725724380204 w_k1_ADM_s106
                                 + 0.0190671703904344 w_k1_ADM_s107
                                 + 0.06704158725158 w_k1_ADM_s110
                                 + 0.0258558186473603 w_k1_ADM_s111
                                 + 0.0411394264629877 w_k1_ADM_s112
                                 + 0.0439879878853486 w_k1_ADM_s113
                                 + 0.0638900144149301 w_k1_ADM_s114
                                 + 0.0235560574493525 w_k1_ADM_s115
                                 + 0.0542917099765212 w_k1_ADM_s116
                                 + 0.0519994496205645 w_k1_ADM_s117
                                 + 0.0317422990665574 w_k1_AXP_s000
                                 + 0.0282973518527261 w_k1_AXP_s001
                                 + 0.0251665812706178 w_k1_AXP_s002
                                 + 0.0381286620911617 w_k1_AXP_s003
                                 + 0.0213121405343735 w_k1_AXP_s004
                                 + 0.0331445395067448 w_k1_AXP_s005
                                 + 0.0587291971818793 w_k1_AXP_s006
                                 + 0.0130749699562732 w_k1_AXP_s007
                                 + 0.026696476332851 w_k1_AXP_s010
                                 + 0.0277823592814365 w_k1_AXP_s011
                                 + 0.0130766258247597 w_k1_AXP_s012
                                 + 0.0193416587819274 w_k1_AXP_s013
                                 + 0.0071914461479622 w_k1_AXP_s014
                                 + 0.0368030813684378 w_k1_AXP_s015
                                 + 0.0166141118692337 w_k1_AXP_s016
                                 + 0.0138291467654965 w_k1_AXP_s017
                                 + 0.022120559053932 w_k1_AXP_s100
                                 + 0.0468717029315524 w_k1_AXP_s101
                                 + 0.0166277522638728 w_k1_AXP_s102
                                 + 0.0391071271197424 w_k1_AXP_s103
                                 + 0.0313860871689049 w_k1_AXP_s104
                                 + 0.029945220542821 w_k1_AXP_s105
                                 + 0.0527543482437069 w_k1_AXP_s106
                                 + 0.0155433533214213 w_k1_AXP_s107
                                 + 0.0646214758685927 w_k1_AXP_s110
                                 + 0.0267001486088679 w_k1_AXP_s111
                                 + 0.0362061146132014 w_k1_AXP_s112
                                 + 0.0452207324311432 w_k1_AXP_s113
                                 + 0.0543476111366719 w_k1_AXP_s114
                                 + 0.0210442744681549 w_k1_AXP_s115
                                 + 0.05221965726643 w_k1_AXP_s116
                                 + 0.0606359321255676 w_k1_AXP_s117
                                 + 0.0285915688548381 w_k1_DIS_s000
                                 + 0.0297621707963259 w_k1_DIS_s001
                                 + 0.0268537606683845 w_k1_DIS_s002
                                 + 0.0376898937615133 w_k1_DIS_s003
                                 + 0.0224025429143756 w_k1_DIS_s004
                                 + 0.0321606951503301 w_k1_DIS_s005
                                 + 0.0573429695087009 w_k1_DIS_s006
                                 + 0.0137355472406881 w_k1_DIS_s007
                                 + 0.0280073232858802 w_k1_DIS_s010
                                 + 0.028709977213497 w_k1_DIS_s011
                                 + 0.0127413016648684 w_k1_DIS_s012
                                 + 0.0180985886948168 w_k1_DIS_s013
                                 + 0.00703918466722885 w_k1_DIS_s014
                                 + 0.0344368820135505 w_k1_DIS_s015
                                 + 0.0186203802403835 w_k1_DIS_s016
                                 + 0.0133396562709872 w_k1_DIS_s017
                                 + 0.0231667226306558 w_k1_DIS_s100
                                 + 0.0483186021929632 w_k1_DIS_s101
                                 + 0.0147398821438436 w_k1_DIS_s102
                                 + 0.0363920036007759 w_k1_DIS_s103
                                 + 0.030040852365222 w_k1_DIS_s104
                                 + 0.0293600007682635 w_k1_DIS_s105
                                 + 0.0553146862794045 w_k1_DIS_s106
                                 + 0.0168396278503671 w_k1_DIS_s107
                                 + 0.0641232524744206 w_k1_DIS_s110
                                 + 0.0285412927141443 w_k1_DIS_s111
                                 + 0.0386274108719174 w_k1_DIS_s112
                                 + 0.0450694276241913 w_k1_DIS_s113
                                 + 0.057945052613319 w_k1_DIS_s114
                                 + 0.0221344800081701 w_k1_DIS_s115
                                 + 0.0548142646183971 w_k1_DIS_s116
                                 + 0.0627397634821109 w_k1_DIS_s117
                                 + 0.02804362237918 w_k1_FDX_s000
                                 + 0.0283402331388305 w_k1_FDX_s001
                                 + 0.0255037126358246 w_k1_FDX_s002
                                 + 0.0369515722099385 w_k1_FDX_s003
                                 + 0.0219617438105642 w_k1_FDX_s004
                                 + 0.030956482467872 w_k1_FDX_s005
                                 + 0.0557037144535336 w_k1_FDX_s006
                                 + 0.0132781778722528 w_k1_FDX_s007
                                 + 0.0275366072377528 w_k1_FDX_s010
                                 + 0.0275269420615979 w_k1_FDX_s011
                                 + 0.0133216175379057 w_k1_FDX_s012
                                 + 0.0182250300359751 w_k1_FDX_s013
                                 + 0.00729156537454465 w_k1_FDX_s014
                                 + 0.0339662373643684 w_k1_FDX_s015
                                 + 0.0166165548434497 w_k1_FDX_s016
                                 + 0.0128392376155643 w_k1_FDX_s017
                                 + 0.023395583450115 w_k1_FDX_s100
                                 + 0.0461639061515224 w_k1_FDX_s101
                                 + 0.0152789098028042 w_k1_FDX_s102
                                 + 0.0379487481294017 w_k1_FDX_s103
                                 + 0.0316845489231219 w_k1_FDX_s104
                                 + 0.0289424897611113 w_k1_FDX_s105
                                 + 0.051118388448466 w_k1_FDX_s106
                                 + 0.0157230273088001 w_k1_FDX_s107
                                 + 0.0659547225799556 w_k1_FDX_s110
                                 + 0.0275359207471082 w_k1_FDX_s111
                                 + 0.0387583014446188 w_k1_FDX_s112
                                 + 0.0453727327399715 w_k1_FDX_s113
                                 + 0.0548418473642029 w_k1_FDX_s114
                                 + 0.0231851127506914 w_k1_FDX_s115
                                 + 0.0503014224695083 w_k1_FDX_s116
                                 + 0.0572534129670759 w_k1_FDX_s117
                                 + 0.0295426074045445 w_k1_IBM_s000
                                 + 0.0288262365279378 w_k1_IBM_s001
                                 + 0.0260694314397293 w_k1_IBM_s002
                                 + 0.0370509997405454 w_k1_IBM_s003
                                 + 0.0229852280762574 w_k1_IBM_s004
                                 + 0.0327346652166841 w_k1_IBM_s005
                                 + 0.0577955183351479 w_k1_IBM_s006
                                 + 0.0136883768441867 w_k1_IBM_s007
                                 + 0.0286990632575319 w_k1_IBM_s010
                                 + 0.0271479284759231 w_k1_IBM_s011
                                 + 0.0129166820772289 w_k1_IBM_s012
                                 + 0.0180370638051959 w_k1_IBM_s013
                                 + 0.0077360703186733 w_k1_IBM_s014
                                 + 0.0348145208998332 w_k1_IBM_s015
                                 + 0.0172679711709246 w_k1_IBM_s016
                                 + 0.0132748072013164 w_k1_IBM_s017
                                 + 0.0244094508737457 w_k1_IBM_s100
                                 + 0.0472823724524065 w_k1_IBM_s101
                                 + 0.0150673033755678 w_k1_IBM_s102
                                 + 0.0389759858002297 w_k1_IBM_s103
                                 + 0.0312228129640705 w_k1_IBM_s104
                                 + 0.0308499856265387 w_k1_IBM_s105
                                 + 0.0515506514036783 w_k1_IBM_s106
                                 + 0.0161700539836925 w_k1_IBM_s107
                                 + 0.0657686645977411 w_k1_IBM_s110
                                 + 0.0264451395807342 w_k1_IBM_s111
                                 + 0.0380852885632672 w_k1_IBM_s112
                                 + 0.0458649406166358 w_k1_IBM_s113
                                 + 0.0570478881687652 w_k1_IBM_s114
                                 + 0.0224075919747361 w_k1_IBM_s115
                                 + 0.0536377150543977 w_k1_IBM_s116
                                 + 0.0609105974850182 w_k1_IBM_s117
                                 + 0.0260113998304912 w_k1_INTC_s000
                                 + 0.0336065862844354 w_k1_INTC_s001
                                 + 0.0281580755107983 w_k1_INTC_s002
                                 + 0.032971837439433 w_k1_INTC_s003
                                 + 0.0211668593727979 w_k1_INTC_s004
                                 + 0.0296111624720724 w_k1_INTC_s005
                                 + 0.0510658009484168 w_k1_INTC_s006
                                 + 0.0144029167898707 w_k1_INTC_s007
                                 + 0.0311113681633367 w_k1_INTC_s010
                                 + 0.0263994524808071 w_k1_INTC_s011
                                 + 0.0119107078476291 w_k1_INTC_s012
                                 + 0.0168042680427929 w_k1_INTC_s013
                                 + 0.00901416890566548 w_k1_INTC_s014
                                 + 0.0328747604779683 w_k1_INTC_s015
                                 + 0.0180010520819854 w_k1_INTC_s016
                                 + 0.0137138914500478 w_k1_INTC_s017
                                 + 0.0218782539342095 w_k1_INTC_s100
                                 + 0.0390283725910225 w_k1_INTC_s101
                                 + 0.0148542712759681 w_k1_INTC_s102
                                 + 0.0389659803828682 w_k1_INTC_s103
                                 + 0.0288899622421301 w_k1_INTC_s104
                                 + 0.0307276829254178 w_k1_INTC_s105
                                 + 0.0557578503814037 w_k1_INTC_s106
                                 + 0.0161625388979627 w_k1_INTC_s107
                                 + 0.0634148450773399 w_k1_INTC_s110
                                 + 0.0252905919350425 w_k1_INTC_s111
                                 + 0.0339632690086151 w_k1_INTC_s112
                                 + 0.0488172234677797 w_k1_INTC_s113
                                 + 0.0484657968062142 w_k1_INTC_s114
                                 + 0.020275139331199 w_k1_INTC_s115
                                 + 0.0517445325112184 w_k1_INTC_s116
                                 + 0.0622355260633072 w_k1_INTC_s117
                                 + 0.0278694816674279 w_k1_KO_s000
                                 + 0.0285635294677599 w_k1_KO_s001
                                 + 0.027664113605915 w_k1_KO_s002
                                 + 0.0382266344975347 w_k1_KO_s003
                                 + 0.0233993603240785 w_k1_KO_s004
                                 + 0.0302532229296151 w_k1_KO_s005
                                 + 0.0508077473948127 w_k1_KO_s006
                                 + 0.0152258616540853 w_k1_KO_s007
                                 + 0.0293177205646255 w_k1_KO_s010
                                 + 0.0284901096348445 w_k1_KO_s011
                                 + 0.0129663210683829 w_k1_KO_s012
                                 + 0.0167795211551892 w_k1_KO_s013
                                 + 0.00705476419678044 w_k1_KO_s014
                                 + 0.0341578810034562 w_k1_KO_s015
                                 + 0.0171002078254111 w_k1_KO_s016
                                 + 0.0121878317853417 w_k1_KO_s017
                                 + 0.0240645671448852 w_k1_KO_s100
                                 + 0.0482372771232705 w_k1_KO_s101
                                 + 0.0138353867978375 w_k1_KO_s102
                                 + 0.0406054003409896 w_k1_KO_s103
                                 + 0.0290965429131054 w_k1_KO_s104
                                 + 0.0294871535138193 w_k1_KO_s105
                                 + 0.0548292516886181 w_k1_KO_s106
                                 + 0.0176274167463777 w_k1_KO_s107
                                 + 0.0622268342764815 w_k1_KO_s110
                                 + 0.0285311845856299 w_k1_KO_s111
                                 + 0.0362809788302026 w_k1_KO_s112
                                 + 0.0425512561786909 w_k1_KO_s113
                                 + 0.0483634044811465 w_k1_KO_s114
                                 + 0.0237494357070709 w_k1_KO_s115
                                 + 0.0542695807425478 w_k1_KO_s116
                                 + 0.0636345846170538 w_k1_KO_s117
                                 + 0.0326390618432191 w_k1_MAT_s000
                                 + 0.0273422372356548 w_k1_MAT_s001
                                 + 0.0235503971431395 w_k1_MAT_s002
                                 + 0.0347123133668113 w_k1_MAT_s003
                                 + 0.0233217036711057 w_k1_MAT_s004
                                 + 0.0330189424474627 w_k1_MAT_s005
                                 + 0.0710400572955074 w_k1_MAT_s006
                                 + 0.0133392269067779 w_k1_MAT_s007
                                 + 0.0362234811807398 w_k1_MAT_s010
                                 + 0.0285333315055655 w_k1_MAT_s011
                                 + 0.0157369012479476 w_k1_MAT_s012
                                 + 0.0210613084251526 w_k1_MAT_s013
                                 + 0.00621133043655096 w_k1_MAT_s014
                                 + 0.0345337843557856 w_k1_MAT_s015
                                 + 0.0149183762181237 w_k1_MAT_s016
                                 + 0.013231061780714 w_k1_MAT_s017
                                 + 0.0281273612789954 w_k1_MAT_s100
                                 + 0.0480508413441002 w_k1_MAT_s101
                                 + 0.0183672709584864 w_k1_MAT_s102
                                 + 0.0352522150480954 w_k1_MAT_s103
                                 + 0.0348418170532228 w_k1_MAT_s104
                                 + 0.0277983613880225 w_k1_MAT_s105
                                 + 0.0451000974455969 w_k1_MAT_s106
                                 + 0.0189232406510816 w_k1_MAT_s107
                                 + 0.0554226556402275 w_k1_MAT_s110
                                 + 0.0267258772695809 w_k1_MAT_s111
                                 + 0.0354202146367746 w_k1_MAT_s112
                                 + 0.0490793413922262 w_k1_MAT_s113
                                 + 0.0562653400754907 w_k1_MAT_s114
                                 + 0.024799843283248 w_k1_MAT_s115
                                 + 0.0468151270218388 w_k1_MAT_s116
                                 + 0.054487476741262 w_k1_MAT_s117
                                 + 0.0278257072216993 w_k1_MSFT_s000
                                 + 0.0291644104019549 w_k1_MSFT_s001
                                 + 0.0273299662443128 w_k1_MSFT_s002
                                 + 0.0378733935765203 w_k1_MSFT_s003
                                 + 0.0233964621272607 w_k1_MSFT_s004
                                 + 0.030387371471838 w_k1_MSFT_s005
                                 + 0.055176436094128 w_k1_MSFT_s006
                                 + 0.012716929082053 w_k1_MSFT_s007
                                 + 0.0288676430517843 w_k1_MSFT_s010
                                 + 0.0256218810865016 w_k1_MSFT_s011
                                 + 0.0112621231639244 w_k1_MSFT_s012
                                 + 0.0184629553175325 w_k1_MSFT_s013
                                 + 0.00732853116924585 w_k1_MSFT_s014
                                 + 0.0333304718100609 w_k1_MSFT_s015
                                 + 0.0159511434684664 w_k1_MSFT_s016
                                 + 0.0141922894150047 w_k1_MSFT_s017
                                 + 0.0217668596752225 w_k1_MSFT_s100
                                 + 0.0453197855333227 w_k1_MSFT_s101
                                 + 0.0147224760055844 w_k1_MSFT_s102
                                 + 0.0426253963536398 w_k1_MSFT_s103
                                 + 0.0290770460084288 w_k1_MSFT_s104
                                 + 0.0318199667396697 w_k1_MSFT_s105
                                 + 0.0502700963102014 w_k1_MSFT_s106
                                 + 0.0176194822961803 w_k1_MSFT_s107
                                 + 0.0619056084453396 w_k1_MSFT_s110
                                 + 0.024920473956972 w_k1_MSFT_s111
                                 + 0.0388503490402427 w_k1_MSFT_s112
                                 + 0.0449417977388905 w_k1_MSFT_s113
                                 + 0.0519142845588416 w_k1_MSFT_s114
                                 + 0.023560579758015 w_k1_MSFT_s115
                                 + 0.0513961242719781 w_k1_MSFT_s116
                                 + 0.0555171887199569 w_k1_MSFT_s117
                                 + 0.0275049603008879 w_k1_PG_s000
                                 + 0.028020367522474 w_k1_PG_s001
                                 + 0.0257782924991646 w_k1_PG_s002
                                 + 0.0370045838260992 w_k1_PG_s003
                                 + 0.0250174135214375 w_k1_PG_s004
                                 + 0.0313772128071375 w_k1_PG_s005
                                 + 0.0521207745966915 w_k1_PG_s006
                                 + 0.0130368099199433 w_k1_PG_s007
                                 + 0.0282948223399719 w_k1_PG_s010
                                 + 0.0262403078444118 w_k1_PG_s011
                                 + 0.014776876984596 w_k1_PG_s012
                                 + 0.0175332021233663 w_k1_PG_s013
                                 + 0.00776183310627574 w_k1_PG_s014
                                 + 0.0347347276425791 w_k1_PG_s015
                                 + 0.0155245555999798 w_k1_PG_s016
                                 + 0.0113522714878117 w_k1_PG_s017
                                 + 0.023927647772654 w_k1_PG_s100
                                 + 0.043602997257516 w_k1_PG_s101
                                 + 0.0149641578094688 w_k1_PG_s102
                                 + 0.0381332828297017 w_k1_PG_s103
                                 + 0.0317475022837075 w_k1_PG_s104
                                 + 0.0280938708368996 w_k1_PG_s105
                                 + 0.0515029333390738 w_k1_PG_s106
                                 + 0.017481482078564 w_k1_PG_s107
                                 + 0.0643868240720249 w_k1_PG_s110
                                 + 0.0262140842831637 w_k1_PG_s111
                                 + 0.0369622975857284 w_k1_PG_s112
                                 + 0.0465947781339825 w_k1_PG_s113
                                 + 0.0558020953275937 w_k1_PG_s114
                                 + 0.0241281617276637 w_k1_PG_s115
                                 + 0.0494999177243002 w_k1_PG_s116
                                 + 0.0616285640576783 w_k1_PG_s117 >= 1.19
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
