import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { arm, defineArm } from "../src/ik/index.js";
import { DreamConfigError } from "../src/index.js";
import { parseXml } from "../src/internal.js";

const metalUrdf = readFileSync(new URL("./fixtures/metal.urdf", import.meta.url), "utf8");
const file = { base: "base_link", flange: "link6", tcp: [0.135, 0, 0], approach: [1, 0, 0], jaw: [0, 1, 0], rest: [0, -1.2, 1.5, 0, 0, 0] };

/** metal.urdf with `from` (which must occur exactly once) replaced by `to`. */
function edit(from: string, to: string): string {
  expect(metalUrdf.split(from).length, from).toBe(2);
  return metalUrdf.replace(from, to);
}

/** defineArm on `urdf` throws a DreamConfigError whose message matches `message`. */
function refuses(urdf: string, message: RegExp, keys: Partial<typeof file> = {}): void {
  expect(() => defineArm({ ...file, ...keys }, urdf)).toThrow(DreamConfigError);
  expect(() => defineArm({ ...file, ...keys }, urdf)).toThrow(message);
}

describe("defineArm refuses a malformed URDF", () => {
  it("still builds metal.urdf, as the built-in Metal arm", () => {
    const custom = defineArm(file, metalUrdf).sideArm("right");
    const builtIn = arm("metal");
    for (const q of [builtIn.data.restPose, [0.3, -1, 1.2, 0.4, -0.5, 1]]) {
      const a = custom.tool(q);
      const b = builtIn.tool(q);
      a.position.forEach((v, i) => expect(Math.abs(v - b.position[i]!)).toBeLessThan(1e-12));
      a.rotation.forEach((v, i) => expect(Math.abs(v - b.rotation[i]!)).toBeLessThan(1e-12));
    }
  });

  it("refuses a loop of joints above the flange instead of walking it forever", () => {
    const urdf = `<robot name="loop"><link name="base"/><link name="a"/><link name="b"/>
      <joint name="ab" type="fixed"><parent link="a"/><child link="b"/></joint>
      <joint name="ba" type="fixed"><parent link="b"/><child link="a"/></joint></robot>`;
    refuses(urdf, /joints above flange 'b' loop back to link 'b'/, { base: "base", flange: "b" });
  });

  it("refuses a link with two parent joints", () => {
    const second = `<joint name="joint6b" type="fixed"><parent link="link4"/><child link="link6"/></joint>\n</robot>`;
    refuses(edit("</robot>", second), /link 'link6' is the child of joints 'joint6' and 'joint6b'/);
  });

  it("refuses an origin or axis that is not 3 finite numbers", () => {
    refuses(edit('xyz="-0.28 0 0"', 'xyz="-0.28 0"'), /joint 'joint3' has origin xyz "-0.28 0", not 3 finite numbers/);
    refuses(edit('xyz="-0.28 0 0"', 'xyz="-0.28 0 0 1"'), /joint 'joint3' has origin xyz/);
    refuses(edit('xyz="-0.28 0 0"', 'xyz="a b c"'), /joint 'joint3' has origin xyz/);
    refuses(edit('xyz="-0.28 0 0"', 'xyz="NaN 0 0"'), /joint 'joint3' has origin xyz/);
    refuses(edit('xyz="-0.28 0 0"', 'xyz=""'), /joint 'joint3' has origin xyz/);
    refuses(edit('xyz="-0.28 0 0" rpy="0 0 0"', 'xyz="-0.28 0 0" rpy="0 Infinity 0"'), /joint 'joint3' has origin rpy/);
    refuses(edit('<axis xyz="1 0 0"/>', '<axis xyz="1 0"/>'), /joint 'joint6' has axis xyz "1 0", not 3 finite numbers/);
  });

  it("refuses limits that are not finite numbers", () => {
    refuses(edit('lower="0" upper="3.14"', 'lower="zero" upper="3.14"'), /joint 'joint3' has limits \["zero","3.14"\], not finite numbers/);
    refuses(edit('lower="0" upper="3.14"', 'lower="0" upper="inf"'), /joint 'joint3' has limits/);
    refuses(edit('lower="0" upper="3.14"', 'lower="" upper="3.14"'), /joint 'joint3' has limits/);
  });

  it("refuses a velocity that is not a finite number of at least 0, and caps a missing or 0 one", () => {
    const limit = '<limit lower="0" upper="3.14" velocity="5" effort="30"/>';
    refuses(edit(limit, limit.replace('velocity="5"', 'velocity="fast"')), /joint 'joint3' has velocity "fast"/);
    refuses(edit(limit, limit.replace('velocity="5"', 'velocity="-1"')), /joint 'joint3' has velocity "-1"/);
    refuses(edit(limit, limit.replace('velocity="5"', 'velocity="Infinity"')), /joint 'joint3' has velocity/);
    for (const to of ['velocity="0"', ""]) {
      expect(defineArm(file, edit(limit, limit.replace('velocity="5"', to))).sideArm("right").data.maxSpeed[2]).toBe(3);
    }
  });

  it("refuses limits with no range 2 degrees inside them", () => {
    refuses(edit('lower="0" upper="3.14"', 'lower="1" upper="1"'), /joint 'joint3' has limits \[1, 1\], an empty range/);
    refuses(edit('lower="0" upper="3.14"', 'lower="1.5" upper="1.55"'), /joint 'joint3' has limits \[1.5, 1.55\], no range left 2 degrees inside them/);
    // 5 degrees wide: 1 degree left between the margins.
    const narrow = edit('lower="0" upper="3.14"', `lower="1.5" upper="${1.5 + (5 * Math.PI) / 180}"`);
    expect(defineArm(file, narrow).sideArm("right").lo[2]).toBeLessThan(defineArm(file, narrow).sideArm("right").hi[2]!);
  });

  it("takes a continuous joint as revolute within [-pi, pi]", () => {
    const model = defineArm(file, edit('name="joint6" type="revolute"', 'name="joint6" type="continuous"')).sideArm("right");
    expect(model.data.limits[5]).toEqual([-Math.PI, Math.PI]);
  });

  it("keeps rest inside the URDF limits, as Python", () => {
    refuses(metalUrdf, /rest puts joint3 at 3.2, outside \[0, 3.14\]/, { rest: [0, -1.2, 3.2, 0, 0, 0] });
    // At a limit, inside the URDF range but not the margined one, as dream.arm_file allows.
    expect(() => defineArm({ ...file, rest: [0, 0, 0, 0, 0, 0] }, metalUrdf)).not.toThrow();
  });
});

describe("parseXml reads the whole URDF strictly, as Python's expat", () => {
  it("refuses an unquoted origin, which a lenient reader skipped, moving the tool 28 cm", () => {
    refuses(edit('<origin xyz="-0.28 0 0"', "<origin xyz=-0.28"), /does not parse at line 31, column 17: expected a quoted value for attribute xyz of <origin>, found "-0.28 rpy/);
  });

  // [what, text, message]; Python's ElementTree refuses each too, except where noted.
  const malformed: [string, string, RegExp][] = [
    ["an unclosed element", '<robot name="r"><link name="a"/>', /line 1, column 1: <robot> is never closed/],
    ["an unterminated tag", '<robot name="r"><link name="a"</robot>', /column 31: expected whitespace, ">" or "\/>" in <link>, found "<\/robot>"/],
    ["a tag cut off by the end", '<robot name="r"><link name="a"', /expected whitespace, ">" or "\/>" in <link>, found the end of the text/],
    ["a mismatched close", '<robot name="r"><link name="a"></joint></robot>', /column 32: <\/joint> closes <link> \(opened at line 1, column 17\)/],
    ["a close of nothing", '<robot name="r"/></link>', /<\/link> closes nothing/],
    ["a stray < in text", '<robot name="r"> a < b </robot>', /column 21: expected a tag name after "<", found " b /],
    // Balanced, so legal XML that Python takes (dropping name="b"); refused as a stray ">" in a tag.
    ["a stray > in a tag", '<robot name="r"><link name="a" > name="b"></link></robot>', /column 42: a bare ">" in the text of <link>/],
    ["a < in a value", '<robot name="a<b"/>', /column 15: "<" in the value of attribute name of <robot>/],
    ["junk after the root", '<robot name="r"/>\njunk', /line 2, column 1: text after the root element, found "junk"/],
    ["a second root", '<robot name="r"/><robot name="s"/>', /a second root element <robot>/],
    ["text before the root", 'junk<robot name="r"/>', /text before the root element/],
    ["a duplicate attribute", '<robot name="r" name="s"/>', /column 17: <robot> has attribute name twice/],
    ["an unquoted value", "<robot name=r/>", /expected a quoted value for attribute name of <robot>, found "r\/>"/],
    ["attributes run together", '<robot name="r"x="1"/>', /expected whitespace, ">" or "\/>" in <robot>, found "x=/],
    ["an unterminated value", '<robot name="r/>', /the value of attribute name of <robot> is never closed/],
    ["an unterminated comment", '<robot name="r"><!-- note </robot>', /a comment is never closed/],
    ["-- inside a comment", '<robot name="r"><!-- a -- b --></robot>', /"--" inside a comment/],
    ["an unknown entity", '<robot name="a &foo; b"/>', /unknown entity &foo;/],
    ["a bare &", '<robot name="a & b"/>', /column 16: "&" starts no entity reference/],
    ["a reference to no character", '<robot name="&#0;"/>', /&#0; names no XML character/],
    ["an unknown entity in text", '<robot name="r"><gazebo>&nbsp;</gazebo></robot>', /unknown entity &nbsp;/],
    ["a control character", '<robot name="r\u0001"/>', /character U\+0001 is not allowed/],
    ["a declaration not at the start", '\n<?xml version="1.0"?><robot name="r"/>', /the XML declaration is not at the start/],
    // Python expands the subset's entities; this reader refuses it rather than misread them.
    ["a DOCTYPE internal subset", '<!DOCTYPE robot [<!ENTITY x "1">]><robot name="&x;"/>', /internal subset/],
    ["no element", "<!-- nothing -->", /it has no root element/],
    ["another root", '<sdf version="1.6"/>', /its root element is <sdf>, not <robot>/],
  ];
  for (const [what, text, message] of malformed) {
    it(`refuses ${what}`, () => {
      expect(() => parseXml(text)).toThrow(DreamConfigError);
      expect(() => parseXml(text)).toThrow(message);
    });
  }

  it("reads well-formed XML: declaration, DOCTYPE, comments, quotes, entities, text, CDATA, self-closing tags", () => {
    const text = [
      "\ufeff<?xml version='1.0' encoding='utf-8'?>\r\n<!DOCTYPE robot SYSTEM \"urdf.dtd\">\r\n<!-- a -->\r\n<?tool x?>",
      "<robot name='say \"hi\"'><!-- in -->",
      '<link name = "a"\n\tfoo\n=\n\'b\' />',
      '<joint name="a &lt;&gt;&amp;&quot;&apos; &#65;&#x42;" xyz="1\n2\t3"/>',
      "<gazebo><robotNamespace>/x &amp; y</robotNamespace><![CDATA[a < b > c]]></gazebo>",
      "</robot >\n<!-- after -->\n",
    ].join("");
    expect(parseXml(text)).toEqual({
      tag: "robot",
      attributes: { name: 'say "hi"' },
      children: [
        { tag: "link", attributes: { name: "a", foo: "b" }, children: [] },
        // Literal tabs and newlines in a value read as spaces, as in XML.
        { tag: "joint", attributes: { name: "a <>&\"' AB", xyz: "1 2 3" }, children: [] },
        { tag: "gazebo", attributes: {}, children: [{ tag: "robotNamespace", attributes: {}, children: [] }] },
      ],
    });
  });

  it("reads metal.urdf element for element", () => {
    const robot = parseXml(metalUrdf);
    expect(robot.children.filter((c) => c.tag === "joint")).toHaveLength(metalUrdf.split("<joint ").length - 1);
    expect(robot.children.filter((c) => c.tag === "link")).toHaveLength(metalUrdf.split("<link ").length - 1);
  });
});
