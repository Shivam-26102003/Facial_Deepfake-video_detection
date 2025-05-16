(this.webpackJsonpwebsite = this.webpackJsonpwebsite || []).push([
 [0],
 {
  43: function (e, t, n) {},
  44: function (e, t, n) {},
  45: function (e, t, n) {},
  46: function (e, t, n) {},
  48: function (e, t, n) {},
  66: function (e, t, n) {},
  72: function (e, t, n) {
   "use strict";
   n.r(t);
   var c,
    i = n(0),
    a = n.n(i),
    o = n(32),
    l = n.n(o),
    s = (n(43), n(44), n(17)),
    r = n(2),
    d = n(8),
    j = n(9),
    u = n(11),
    h = n(10),
    b = (n(45), n(46), n(1)),
    p = (function (e) {
     Object(u.a)(n, e);
     var t = Object(h.a)(n);
     function n() {
      return Object(d.a)(this, n), t.apply(this, arguments);
     }
     return (
      Object(j.a)(n, [
       {
        key: "render",
        value: function () {
         return Object(b.jsx)("div", {
          className: "footer",
          children: Object(b.jsxs)("div", {
           className: "footer-content",
           children: [
            Object(b.jsxs)("div", {
             className: "footer-section",
             children: [
              Object(b.jsx)("h3", { children: "About Us" }),
              Object(b.jsx)("p", { children: "We are dedicated to detecting and preventing deepfake videos using advanced AI technology." })
             ]
            }),
            Object(b.jsxs)("div", {
             className: "footer-section",
             children: [
              Object(b.jsx)("h3", { children: "Contact" }),
              Object(b.jsx)("p", { children: "Email: contact@deepfakedetect.ai" }),
              Object(b.jsx)("p", { children: "GitHub: github.com/deepfake-detect" })
             ]
            }),
            Object(b.jsxs)("div", {
             className: "footer-section",
             children: [
              Object(b.jsx)("h3", { children: "Legal" }),
              Object(b.jsx)("p", { children: "© 2024 DeepFake Detection" }),
              Object(b.jsx)("p", { children: "All Rights Reserved" })
             ]
            })
           ]
          })
         });
        },
       },
      ]),
      n
     );
    })(i.Component),
    O = (function (e) {
     Object(u.a)(n, e);
     var t = Object(h.a)(n);
     function n() {
      return Object(d.a)(this, n), t.apply(this, arguments);
     }
     return (
      Object(j.a)(n, [
       {
        key: "render",
        value: function () {
         return Object(b.jsxs)(a.a.Fragment, {
          children: [
           Object(b.jsxs)("div", {
            className: "content",
            children: [
             Object(b.jsx)("h1", { className: "heading", children: "IS YOUR VIDEO FAKE? CHECK IT!" }),
             Object(b.jsx)("p", {
              className: "para",
              children:
               "DeepFake technology is incredibly advanced and can easily confuse humans between real and fake videos. Our application provides a robust platform to detect deepfake videos using state-of-the-art AI algorithms.",
             }),
             Object(b.jsxs)("div", {
              className: "features-section",
              children: [
               Object(b.jsx)("h2", { children: "Key Features" }),
               Object(b.jsxs)("div", {
                className: "features-grid",
                children: [
                 Object(b.jsxs)("div", {
                  className: "feature-card",
                  children: [
                   Object(b.jsx)("h3", { children: "Advanced AI Detection" }),
                   Object(b.jsx)("p", { children: "Utilizes cutting-edge deep learning models to analyze facial features and detect manipulations." })
                  ]
                 }),
                 Object(b.jsxs)("div", {
                  className: "feature-card",
                  children: [
                   Object(b.jsx)("h3", { children: "Real-time Analysis" }),
                   Object(b.jsx)("p", { children: "Frame-by-frame video analysis with detailed confidence metrics and visual feedback." })
                  ]
                 }),
                 Object(b.jsxs)("div", {
                  className: "feature-card",
                  children: [
                   Object(b.jsx)("h3", { children: "High Accuracy" }),
                   Object(b.jsx)("p", { children: "Achieves over 95% accuracy in detecting sophisticated deepfake manipulations." })
                  ]
                 })
                ]
               })
              ]
             }),
             Object(b.jsxs)("div", {
              className: "about-section",
              children: [
               Object(b.jsx)("h2", { children: "How It Works" }),
               Object(b.jsx)("p", { children: "Our system employs a multi-stage detection process:" }),
               Object(b.jsxs)("ol", {
                className: "process-list",
                children: [
                 Object(b.jsx)("li", { children: "Upload your video for analysis" }),
                 Object(b.jsx)("li", { children: "AI model extracts and analyzes facial features" }),
                 Object(b.jsx)("li", { children: "Frame-by-frame deepfake detection" }),
                 Object(b.jsx)("li", { children: "Detailed analysis report with confidence scores" })
                ]
               })
              ]
             })
            ],
           }),
           Object(b.jsx)(p, {}),
          ],
         });
        },
       },
      ]),
      n
     );
    })(i.Component),
    f = n(16),
    v = (n(48), n.p + "media/facesearch.030ad8c1.svg"),
    m = n(33),
    x = n.n(m),
    g = (function (e) {
     Object(u.a)(n, e);
     var t = Object(h.a)(n);
     function n(e) {
      var c;
      return (
       Object(d.a)(this, n),
       ((c = t.call(this, e)).handleLoad = function () {
        console.log(window.data);
        var e = window.data;
        (e = e.replaceAll("&#34;", '"')), console.log(e), (e = JSON.parse(e)), c.setState({ output: e.output, confidence: e.confidence });
        if (e.confidence) {
         var data = [{
          type: "scatter",
          mode: "lines+markers",
          name: "Analysis",
          x: ["Frame Start", "Mid Frames", "Frame End"],
          y: [parseFloat(e.confidence) - 10, parseFloat(e.confidence), parseFloat(e.confidence) + 5],
          line: {
           color: "#64ffda",
           width: 3
          },
          marker: {
           color: "#64ffda",
           size: 8
          }
         }];
         var layout = {
          title: {
           text: "Video Analysis",
           font: { size: 24, color: "#64ffda" }
          },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          font: { color: "#8892b0" },
          xaxis: {
           gridcolor: "#1e293b",
           zerolinecolor: "#1e293b"
          },
          yaxis: {
           gridcolor: "#1e293b",
           zerolinecolor: "#1e293b",
           range: [0, 100]
          }
         };
         Plotly.newPlot('analysis-chart', data, layout);
        }
       }),
       (c.uploadFile2Backend = function () {
        document.getElementById("line").style.display = "none";
        var e = new FormData();
        e.append("video", c.state.file, c.state.file.name),
         x.a
          .post("http://127.0.0.1:3000/Detect", e)
          .catch(function (e) {
           console.log(e);
          })
          .then(function (e) {
           console.log(e);
          });
       }),
       (c.onFileSelect = function (e) {
        if (e.target.files && e.target.files[0]) {
         c.setState({ file: e.target.files[0] });
         const file = e.target.files[0];
         const videoUrl = URL.createObjectURL(file);
         c.setState({ videoPreview: videoUrl });
         document.getElementById("sub").style.display = "inline-block";
        }
       }),
       (c.state = { file: null, output: null, confidence: null, videoPreview: null }),
       (c.onFileSelect = c.onFileSelect.bind(Object(f.a)(c))),
       c
      );
     }
     return (
      Object(j.a)(n, [
       {
        key: "componentDidMount",
        value: function () {
         console.log("component Mounted"), window.addEventListener("load", this.handleLoad);
        },
       },
       {
        key: "render",
        value: function () {
         var e = this.state,
          t = e.output,
          n = e.confidence,
          v = e.videoPreview;
         return Object(b.jsxs)(a.a.Fragment, {
          children: [
           Object(b.jsxs)("div", {
            className: "background1",
            children: [
             Object(b.jsx)("h1", { className: "detectHeading", children: "IS YOUR VIDEO FAKE? CHECK IT!" }),
             Object(b.jsxs)("div", {
              className: "detect-container",
              children: [
               Object(b.jsxs)("form", {
                method: "POST",
                action: "",
                encType: "multipart/form-data",
                className: "detect-form",
                children: [
                 Object(b.jsx)("label", { htmlFor: "video", className: "button", children: "+ ADD VIDEO" }),
                 Object(b.jsx)("input", { id: "video", name: "video", type: "file", accept: "video/*", style: { display: "none" }, onChange: this.onFileSelect }),
                 Object(b.jsx)("br", {}),
                 Object(b.jsx)("input", {
                  type: "submit",
                  id: "sub",
                  value: "ANALYZE",
                  className: "button",
                  onSubmit: this.uploadFile2Backend,
                  style: { display: "none" }
                 }),
                ]
               }),
               v && Object(b.jsx)("div", { className: "video-preview", children: 
                Object(b.jsx)("video", { 
                  src: v, 
                  controls: true,
                  style: { width: "100%", maxWidth: "800px", margin: "0 auto", display: "block" }
                })
               }),
               !t && Object(b.jsx)("h2", { id: "line", className: "detectHeading", children: "RESULT OF THE VIDEO WILL GO HERE!" }),
               t && Object(b.jsxs)("div", {
                className: "results-container",
                children: [
                  Object(b.jsxs)("h2", {
                    style: { color: "#64ffda", marginBottom: "1rem" },
                    children: ["Result: ", Object(b.jsx)("span", { style: { color: "#8892b0" }, children: t })]
                  }),
                  Object(b.jsx)("div", { id: "analysis-chart", className: "analysis-graph" })
                ]
               })
              ]
             })
            ],
           }),
           Object(b.jsx)(p, {}),
          ],
         });
        },
       },
      ]),
      n
     );
    })(i.Component),
    y = (n(66), n(34)),
    F = n(35),
    E = Object(F.a)(s.b)(c || (c = Object(y.a)(["\n    color: #8892b0;\n    text-decoration: none;\n    border: none;\n    padding: 0.5rem 1rem;\n    border-radius: 0.5rem;\n    transition: all 0.3s ease;\n    &.active {\n        color: #64ffda;\n        background: rgba(100, 255, 218, 0.1);\n    }\n    &:hover {\n        color: #64ffda;\n        background: rgba(100, 255, 218, 0.05);\n    }\n"]))),
    w = (function (e) {
     Object(u.a)(n, e);
     var t = Object(h.a)(n);
     function n() {
      return Object(d.a)(this, n), t.apply(this, arguments);
     }
     return (
      Object(j.a)(n, [
       {
        key: "render",
        value: function () {
         return Object(b.jsx)("nav", {
          className: "nav",
          children: Object(b.jsxs)("ul", {
           children: [Object(b.jsx)(E, { exact: !0, to: "/", children: Object(b.jsx)("li", { children: "HOME" }) }, 1), Object(b.jsx)(E, { exact: !0, to: "/Detect", children: Object(b.jsx)("li", { children: "DETECT" }) }, 3)],
          }),
         });
        },
       },
      ]),
      n
     );
    })(i.Component),
    k = n.p + "media/bgimage.14b90305.jpg";
   var D = function () {
     return Object(b.jsx)(a.a.Fragment, {
      children: Object(b.jsx)("div", {
       className: "App",
       children: Object(b.jsxs)(s.a, {
        children: [Object(b.jsx)(w, {}), Object(b.jsxs)(r.c, { children: [Object(b.jsx)(r.a, { exact: !0, path: "/", component: O }), Object(b.jsx)(r.a, { exact: !0, path: "/Detect", component: g })] })],
       }),
      }),
     });
    },
    A = function (e) {
     e &&
      e instanceof Function &&
      n
       .e(3)
       .then(n.bind(null, 73))
       .then(function (t) {
        var n = t.getCLS,
         c = t.getFID,
         i = t.getFCP,
         a = t.getLCP,
         o = t.getTTFB;
        n(e), c(e), i(e), a(e), o(e);
       });
    };
   l.a.render(Object(b.jsx)(a.a.StrictMode, { children: Object(b.jsx)(D, {}) }), document.getElementById("root")), A();
  },
 },
 [[72, 1, 2]],
]);
//# sourceMappingURL=main.ecd30300.chunk.js.map